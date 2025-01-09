from openai import OpenAI
import asyncio, json, time, uuid, logging, os, threading
from pathlib import Path
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
from dotenv import load_dotenv
from dataclasses import dataclass, asdict, field
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, 
    cors_allowed_origins="*",
    async_mode="threading",
    logger=True,
    engineio_logger=True,
    allow_unsafe_werkzeug=True
)

# Constants for DISC and Phases
DISC_TYPES = ["D", "I", "S", "C"]

PHASE_DISTRIBUTION = {
    "INITIATION": {
        "rounds": [1, 2],
        "purpose": "Setting foundation and scope",
        "behaviors": ["Define scope", "Establish framework"]
    },
    "EXPLORATION": {
        "rounds": [3, 4, 5],
        "purpose": "Broadening perspectives",
        "behaviors": ["Generate viewpoints", "Share experiences"]
    },
    "FOCUS_EMERGENCE": {
        "rounds": [6, 7],
        "purpose": "Identifying key themes",
        "behaviors": ["Highlight patterns", "Select focus areas"]
    },
    "DEEP_DIVE": {
        "rounds": [8, 9, 10],
        "purpose": "Thorough examination",
        "behaviors": ["Detailed analysis", "Evidence presentation"]
    },
    "CONVERGENCE": {
        "rounds": [11, 12],
        "purpose": "Pulling insights together",
        "behaviors": ["Connect findings", "Bridge viewpoints"]
    },
    "SYNTHESIS": {
        "rounds": [13, 14],
        "purpose": "Integrating insights",
        "behaviors": ["Connect themes", "Frame understanding"]
    },
    "CLOSURE": {
        "rounds": [15],
        "purpose": "Ensuring takeaways",
        "behaviors": ["Summarize learnings", "Frame implications"]
    }
}

# Helper Functions
def safe_emit(event, data):
    try:
        socketio.emit(event, data)
        logging.info(f"Emitted {event}: {data}")
    except Exception as e:
        logging.error(f"Socket emission failed: {e}")

def get_phase_for_round(round_num: int) -> str:
    for phase, info in PHASE_DISTRIBUTION.items():
        if round_num in info["rounds"]:
            return phase
    return "CLOSURE"

# Socket Events
@socketio.on('connect')
def connect():
    logging.info('Client connected')

@socketio.on('disconnect')
def disconnect():
    logging.info('Client disconnected')

# Data Classes
@dataclass
class Message:
    agent: str
    content: str
    timestamp: float
    type: str

@dataclass
class Subtopic:
    id: str
    topic: str
    agents: List[str]
    messages: List[Message]
    summary: Optional[str]
    start_time: float
    end_time: Optional[float]
    agent_traits: dict = field(default_factory=dict)

@dataclass
class Discussion:
    main_topic: str
    subtopics: List[Subtopic]
    current_subtopic_id: Optional[str]

# Main Discussion Manager Class
class DiscussionManager:
    def __init__(self, api_key: str, storage_path: str = "discussions"):
        self.client = OpenAI(api_key=api_key)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.discussion = None
        self.current_topic_index = 0
        self.is_processing = False
        self.topics = []
        self.current_round = 0
        self.total_rounds = 2
        self.load_topics_from_env()

    def load_topics_from_env(self):
        """Load topics from environment variables."""
        load_dotenv(override=True)
        env_topics = os.getenv("TOPICS", "").split(",")
        self.topics = [topic.strip() for topic in env_topics if topic.strip()]
        if self.topics:
            logging.info(f"Loaded topics from env: {self.topics}")
        else:
            logging.warning("No topics found in environment variables")

    async def start_main_discussion(self):
        while True:
            if not self.topics:
                self.load_topics_from_env()
                await asyncio.sleep(10)
                continue

            if self.current_topic_index >= len(self.topics):
                self.current_topic_index = 0
                await asyncio.sleep(10)
                self.load_topics_from_env()
                continue

            main_topic = self.topics[self.current_topic_index]
            if not self.is_processing:
                self.is_processing = True
                logging.info(f"Starting discussion on: {main_topic}")
                self.discussion = Discussion(main_topic=main_topic, subtopics=[], current_subtopic_id=None)
                
                start_time = time.time()
                time_limit = 7200
                is_in_discussion = False
                
                while True:
                    current_time = time.time()
                    if not is_in_discussion and (current_time - start_time) >= time_limit:
                        break
                        
                    is_in_discussion = True
                    await self._run_discussion()
                    is_in_discussion = False
                    
                await self._finalize_discussion()
                self.is_processing = False
                self.current_topic_index += 1
                self.load_topics_from_env()

    async def _generate_custom_agents(self, subtopic: str) -> List[dict]:
        system_prompt = """You are a JSON generator that creates balanced agent profiles with DISC types. Generate exactly 4 agents, each with a different DISC type."""
        
        user_prompt = f"""
        Create 4 specialized agents for discussing: {subtopic}
        
        Each agent must have:
        1. Unique ID with prefix OB_3LETTER_4RANDOMINT(e.g., "OB-XYZ-5623")
        2. DISC Type (one each of D, I, S, C)
        3. Expertise (2-3 areas specific to {subtopic})
        4. Core personality traits aligned with DISC type
        5. Natural communication style
        
        Return in exact JSON format:
        {{
            "agents": [
                {{
                    "id": "OB-001",
                    "disc_type": "D",
                    "expertise": ["area1", "area2"],
                    "personality": {{
                        "core_trait": "trait",
                        "traits": ["trait1", "trait2"]
                    }}
                }}
            ]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            agents = result.get("agents", [])
            
            # Ensure DISC balance
            disc_types_used = [agent.get("disc_type") for agent in agents]
            if len(set(disc_types_used)) != 4:
                for i, agent in enumerate(agents):
                    agent["disc_type"] = DISC_TYPES[i]
            
            return agents
            
        except Exception as e:
            logging.error(f"Error generating agents: {e}")
            return self._generate_fallback_agents(subtopic)

    def _get_disc_behavior(self, disc_type: str, phase: str) -> str:
        behaviors = {
            "D": {
                "INITIATION": "set clear direction and drive for concrete scope",
                "EXPLORATION": "push for diverse perspectives and quick insights",
                "FOCUS_EMERGENCE": "identify strongest threads to pursue",
                "DEEP_DIVE": "drive toward concrete findings",
                "CONVERGENCE": "push for clear outcomes",
                "SYNTHESIS": "crystallize key points",
                "CLOSURE": "establish clear takeaways"
            },
            "I": {
                "INITIATION": "generate excitement and possibilities",
                "EXPLORATION": "create engaging connections between ideas",
                "FOCUS_EMERGENCE": "maintain energy while helping focus",
                "DEEP_DIVE": "keep engagement during analysis",
                "CONVERGENCE": "bridge different viewpoints",
                "SYNTHESIS": "create compelling narrative",
                "CLOSURE": "end with positive energy"
            },
            "S": {
                "INITIATION": "ensure everyone's voice is heard",
                "EXPLORATION": "support thorough consideration",
                "FOCUS_EMERGENCE": "help build consensus",
                "DEEP_DIVE": "maintain steady progress",
                "CONVERGENCE": "harmonize different views",
                "SYNTHESIS": "ensure complete integration",
                "CLOSURE": "confirm group satisfaction"
            },
            "C": {
                "INITIATION": "establish precise framework",
                "EXPLORATION": "analyze options systematically",
                "FOCUS_EMERGENCE": "evaluate patterns critically",
                "DEEP_DIVE": "ensure thorough analysis",
                "CONVERGENCE": "verify logical consistency",
                "SYNTHESIS": "validate all conclusions",
                "CLOSURE": "document precise outcomes"
            }
        }
        return behaviors.get(disc_type, {}).get(phase, "contribute based on discussion needs")

    async def _generate_response(self, agent_info: dict, subtopic: Subtopic) -> str:
        current_phase = get_phase_for_round(self.current_round)
        
        # Organize messages by rounds
        messages_by_round = {}
        current_round_messages = []
        
        for msg in subtopic.messages:
            msg_round = (len(subtopic.messages) // len(subtopic.agents)) + 1
            
            if msg_round == self.current_round:
                current_round_messages.append(msg)
            else:
                if msg_round not in messages_by_round:
                    messages_by_round[msg_round] = []
                messages_by_round[msg_round].append(msg)

        # Track questions and key points with agent references
        current_questions = []
        recent_key_points = []
        
        # Get agent profiles for context
        agent_profiles = {
            agent_id: {
                "disc_type": traits["disc_type"],
                "expertise": traits["expertise"],
                "core_trait": traits["core_trait"],
                "last_referenced": 0  # Track when this agent was last referenced
            }
            for agent_id, traits in subtopic.agent_traits.items()
        }

        system_prompt = f"""
        You are an expert participating in a structured group discussion about '{subtopic.topic}' 
        as part of exploring '{self.discussion.main_topic}'.

        Your Profile:
        - Agent ID: {agent_info['id']}
        - DISC Type: {agent_info['disc_type']}
        - Expertise: {', '.join(agent_info['expertise'])}
        - Core Trait: {agent_info['personality']['core_trait']}

        Other Participants:
        {chr(10).join(f"- {aid} ({prof['disc_type']} type, Expert in: {', '.join(prof['expertise'])})" for aid, prof in agent_profiles.items() if aid != agent_info['id'])}

        Response Guidelines for Unique Perspectives:
        - Express distinct viewpoints based on your expertise and DISC type
        - Challenge assumptions when appropriate
        - Bring in novel examples and analogies
        - Connect ideas in unexpected but relevant ways
        - Avoid simply agreeing or disagreeing - add new dimensions
        
Response Guidelines for Agent References:
        - Reference other agents naturally and selectively:
          * Direct responses: Reference the agent you're directly responding to
          * Building on ideas: Reference 1-2 agents whose points you're connecting
          * New points: No need to reference others unless building on their ideas
        - Vary between:
          * Direct references: "Building on [AgentID]'s point..."
          * Implicit references: "This approach could complement the sustainability concerns raised earlier..."
          * Independent statements: Make your own points without references when appropriate
        
        Current Phase: {current_phase}
        Phase Purpose: {PHASE_DISTRIBUTION[current_phase]['purpose']}
        """

        try:
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add previous rounds context with reference tracking
            for round_num in sorted(messages_by_round.keys()):
                round_messages = messages_by_round[round_num]
                round_phase = get_phase_for_round(round_num)
                
                round_context = []
                for msg in round_messages:
                    agent_disc = agent_profiles[msg.agent]["disc_type"]
                    agent_expertise = ', '.join(agent_profiles[msg.agent]["expertise"])
                    round_context.append(f"{msg.agent} ({agent_disc} type, Expert in: {agent_expertise}): {msg.content}")
                    
                    # Update last reference tracking
                    agent_profiles[msg.agent]["last_referenced"] = round_num
                
                messages.append({
                    "role": "assistant",
                    "content": f"""Round {round_num} Discussion (Phase: {round_phase}):
                    {PHASE_DISTRIBUTION[round_phase]['purpose']}
                    
                    {chr(10).join(round_context)}
                    """
                })
            
            # Add current round context
            if current_round_messages:
                current_context = []
                for msg in current_round_messages:
                    if '?' in msg.content:
                        current_questions.append(f"{msg.agent}: {msg.content}")
                    recent_key_points.append(f"{msg.agent}'s point: {msg.content}")
                    
                    agent_disc = agent_profiles[msg.agent]["disc_type"]
                    agent_expertise = ', '.join(agent_profiles[msg.agent]["expertise"])
                    current_context.append(f"{msg.agent} ({agent_disc} type, Expert in: {agent_expertise}): {msg.content}")
                    
                    # Update last reference tracking
                    agent_profiles[msg.agent]["last_referenced"] = self.current_round

                messages.append({
                    "role": "assistant",
                    "content": f"""Current Round {self.current_round} Discussion:
                    Phase: {current_phase}
                    Purpose: {PHASE_DISTRIBUTION[current_phase]['purpose']}
                    
                    {chr(10).join(current_context)}
                    """
                })
            
            # Identify agents to potentially reference based on recency and relevance
            recent_contributors = [
                agent_id for agent_id, profile in agent_profiles.items()
                if profile["last_referenced"] >= (self.current_round - 1)
                and agent_id != agent_info['id']
            ]
            
            assistant_prompt = f"""
            As {agent_info['id']}, engage in the discussion considering:

            Current State:
            - Active Questions: {json.dumps(current_questions) if current_questions else "None pending"}
            - Recent Key Points: {json.dumps(recent_key_points[-3:]) if recent_key_points else "Starting discussion"}
            - Recent Contributors: {json.dumps(recent_contributors) if recent_contributors else "None"}

            Reference Guidelines:
            1. If responding to a specific point: Reference that agent directly
            2. If building on multiple ideas: Reference up to 2 relevant agents
            3. If making a new point: No need for references
            4. If synthesizing: Briefly acknowledge relevant contributors
            5. Avoid mentioning other agents in every response. Sometimes focus solely on your own unique thoughts and perspectives.

            Choose your response approach:
            1. Analytical: Evaluate previous points
            2. Supportive: Build upon others' arguments
            3. Challenging: Present alternative viewpoints
            4. Synthesizing: Connect different perspectives
            5. Decisive: Make clear statements
            6. Questioning: Ask if critical information is missing

            Remember:
            - Not to respond in more than 100-200 words
            - References should feel natural, not forced
            - Vary between direct references, implicit references, and independent statements
            - Focus on advancing the {PHASE_DISTRIBUTION[current_phase]['purpose']} objective
            """ 
            
            messages.append({"role": "user", "content": assistant_prompt})

            # Configure parameters for diverse and unique responses
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.95,    # Higher temperature for more creative responses
                top_p=0.9,          # Allow more diverse token selection
                frequency_penalty=1.8,  # Strongly discourage repetition of ideas
                presence_penalty=1.6    # Encourage coverage of new topics
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "Error generating response."

    async def _run_agent_discussion(self, subtopic: Subtopic, custom_agents: List[dict]):
        self.current_round = 1
        
        for _ in range(self.total_rounds):
            for agent_info in custom_agents:
                response = await self._generate_response(agent_info, subtopic)
                
                message = Message(
                    agent=agent_info["id"],
                    content=response,
                    timestamp=time.time(),
                    type="chat"
                )
                subtopic.messages.append(message)
                
                safe_emit("current_topic_stream", {
                    "message": asdict(message),
                    "history": [asdict(msg) for msg in subtopic.messages]
                })
                
                self.current_round += 1
                await asyncio.sleep(3)

    async def _run_head_agent(self, subtopic: Subtopic):
        messages = "\n".join([f"{m.agent}: {m.content}" for m in subtopic.messages])
        prompt = f"""
        Analyze and summarize the discussion about "{subtopic.topic}" in chronological order.
        Include:
        1. Participant Analysis
        2. Discussion Overview
        3. Chronological Development
        4. Notable Contributions
        5. Key Conclusions
        6. Action Items
        7. Recommended Next Topics

        Previous discussion:
        {messages}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a discussion coordinator summarizing key points and action items."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )
        
        summary = response.choices[0].message.content.strip()
        subtopic.summary = summary
        safe_emit("current_topic_stream", {
            "summary": summary,
            "history": [asdict(msg) for msg in subtopic.messages]
        })

    async def _run_discussion(self):
        subtopic = await self._get_next_subtopic()
        custom_agents = await self._generate_custom_agents(subtopic)
        
        new_subtopic = Subtopic(
            id=str(uuid.uuid4()),
            topic=subtopic,
            agents=[agent["id"] for agent in custom_agents],
            messages=[],
            summary=None,
            start_time=time.time(),
            end_time=None,
            agent_traits={
                agent["id"]: {
                    "core_trait": agent["personality"]["core_trait"],
                    "traits": agent["personality"]["traits"],
                    "expertise": agent["expertise"],
                    "disc_type": agent["disc_type"]
                } for agent in custom_agents
            }
        )
        
        self.discussion.subtopics.append(new_subtopic)
        self.discussion.current_subtopic_id = new_subtopic.id
        
        safe_emit("current_topic_stream", {
            "subtopic": new_subtopic.topic,
            "history": []
        })

        await self._run_agent_discussion(new_subtopic, custom_agents)
        await self._run_head_agent(new_subtopic)
        new_subtopic.end_time = time.time()
        self._save_discussion()

    async def _get_next_subtopic(self) -> str:
        subtopic_types = {
            "debate": [
                "Controversial aspects of",
                "Opposing viewpoints on",
                "Critical debate:",
                "Challenging the assumptions about"
            ],
            "evaluation": [
                "Assessment of",
                "Critical analysis:",
                "Evaluating the impact of",
                "Performance review:"
            ],
            "comparison": [
                "Comparing different approaches to",
                "Contrasting perspectives on",
                "Traditional vs modern views on",
                "Competitive analysis:"
            ],
            "impact_analysis": [
                "Long-term implications of",
                "Societal impact of",
                "Future scenarios for",
                "Ripple effects of"
            ],
            "deep_dive": [
                "Technical deep-dive:",
                "Detailed examination of",
                "Core mechanisms of",
                "Understanding the fundamentals of"
            ],
            "synthesis": [
                "Integrating multiple perspectives on",
                "Bridging the gap between",
                "Unified approach to",
                "Synthesizing research on"
            ],
            "practical": [
                "Real-world applications of",
                "Implementation strategies for",
                "Best practices in",
                "Practical solutions for"
            ],
            "innovation": [
                "Emerging trends in",
                "Next-generation approaches to",
                "Revolutionary changes in",
                "Breakthrough developments in"
            ]
        }

        prompt = f"""
        Main topic: "{self.discussion.main_topic}"
        Previous subtopics: "{', '.join(s.topic for s in self.discussion.subtopics)}"
        
        Example of diverse subtopic types using "Artificial Intelligence" as a sample main topic:

        DEBATE subtopics:
        - "Debating AI consciousness: Can machines truly be self-aware?"
        - "Challenging the assumption of AI replacing human jobs"
        - "AI rights: Should advanced AI systems have legal protections?"

        EVALUATION subtopics:
        - "Measuring the effectiveness of AI in healthcare diagnostics"
        - "Assessment of current AI safety protocols"
        - "Performance analysis of AI vs human decision-making"

        COMPARISON subtopics:
        - "Rule-based vs. Neural Network approaches in AI"
        - "Comparing AI adoption across different industries"
        - "Traditional algorithms vs. AI solutions in cybersecurity"

        IMPACT ANALYSIS subtopics:
        - "Long-term implications of AI on human creativity"
        - "Societal ripple effects of autonomous vehicles"
        - "Environmental impact of AI computing infrastructure"

        DEEP DIVE subtopics:
        - "Technical deep-dive: Understanding transformer architectures"
        - "Examining bias in machine learning datasets"
        - "Core mechanisms of reinforcement learning"

        SYNTHESIS subtopics:
        - "Integrating AI with traditional business processes"
        - "Bridging AI capabilities with human expertise"
        - "Unified framework for ethical AI development"

        PRACTICAL subtopics:
        - "Implementation strategies for AI in small businesses"
        - "Best practices in AI model deployment"
        - "Real-world applications of federated learning"

        INNOVATION subtopics:
        - "Emerging trends in multimodal AI systems"
        - "Next-generation approaches to AI training"
        - "Breakthrough developments in quantum AI"

        Using these patterns as inspiration, generate the next subtopic for the current main topic.
        Consider:
        - Previous subtopics to avoid repetition
        - Natural progression of discussion
        - Engagement level and complexity
        - Balance between theoretical and practical aspects

        Consider the discussion flow and vary the type of subtopic from previous ones.
        Ensure the subtopic:
        - Is specific and focused
        - Builds naturally from previous subtopics
        - Offers a fresh perspective or approach
        - Maintains engagement through variety
        - Avoids redundancy with previous subtopics

        Return only the subtopic as a concise phrase.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
    You are an expert discussion coordinator specializing in dynamic topic progression.

    Your expertise includes:
    - Crafting engaging discussion flows
    - Balancing different types of analytical approaches
    - Maintaining topic coherence while exploring diverse angles
    - Creating natural progression between subtopics
    - Varying discussion formats to maintain engagement

    Format your response as a single, specific subtopic phrase.
    """},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7,
            stop=["\n", ".", "Next:", "Topic:"]
        )
        
        # Track the subtopic type for future reference
        subtopic = response.choices[0].message.content.strip()
        
        # Store the subtopic type if needed for future use
        for type_name, prefixes in subtopic_types.items():
            if any(subtopic.startswith(prefix) for prefix in prefixes):
                if not hasattr(self, 'subtopic_types_used'):
                    self.subtopic_types_used = []
                self.subtopic_types_used.append(type_name)
                break
                
        return subtopic

    def _generate_fallback_agents(self, subtopic: str) -> List[dict]:
        """Generate fallback agents if the main generation fails."""
        return [
            {
                "id": f"AG-{i+1}",
                "disc_type": disc_type,
                "expertise": [subtopic],
                "personality": {
                    "core_trait": "analytical",
                    "traits": ["logical", "methodical"]
                }
            }
            for i, disc_type in enumerate(DISC_TYPES)
        ]

    def _save_discussion(self):
        """Save current discussion state to file."""
        discussion_data = asdict(self.discussion)
        with open(self.storage_path / "discussion.json", "w") as f:
            json.dump(discussion_data, f, indent=2)

    async def _finalize_discussion(self):
        """Finalize and save discussion outcomes."""
        objectives_file = self.storage_path / "objectives.json"
        objectives = []
        
        if objectives_file.exists():
            with open(objectives_file) as f:
                objectives = json.load(f)
                
        subtopics_with_traits = []
        for subtopic in self.discussion.subtopics:
            subtopic_data = asdict(subtopic)
            subtopics_with_traits.append(subtopic_data)
        
        objectives.append({
            "main_topic": self.discussion.main_topic,
            "subtopics": subtopics_with_traits
        })
        
        with open(objectives_file, "w") as f:
            json.dump(objectives, f, indent=2)

# Flask Routes
@app.route("/discussions")
def get_discussions():
    """Get all discussions."""
    file_path = Path("discussions/discussion.json")
    if not file_path.exists():
        return jsonify([])
    with open(file_path) as f:
        discussion = json.load(f)
    return jsonify([{
        "id": s["id"],
        "topic": s["topic"],
        "agents": s["agents"],
        "agent_traits": s.get("agent_traits", {}),
        "summary": s["summary"]
    } for s in discussion["subtopics"]])

@app.route("/discussions/<subtopic_id>")
def get_subtopic_details(subtopic_id):
    """Get details for a specific subtopic."""
    file_path = Path("discussions/discussion.json")
    if not file_path.exists():
        return jsonify({"error": "No discussions found"}), 404
        
    with open(file_path) as f:
        discussion = json.load(f)
        for subtopic in discussion["subtopics"]:
            if subtopic["id"] == subtopic_id:
                return jsonify({
                    "topic": subtopic["topic"],
                    "agents": subtopic["agents"],
                    "agent_traits": subtopic.get("agent_traits", {}),
                    "messages": subtopic["messages"],
                    "summary": subtopic["summary"]
                })
                
    return jsonify({"error": "Subtopic not found"}), 404

@app.route("/objectives")
def get_objectives():
    """Get all discussion objectives."""
    file_path = Path("discussions/objectives.json")
    if not file_path.exists():
        return jsonify([])
    with open(file_path) as f:
        data = json.load(f)
        for discussion in data:
            for subtopic in discussion["subtopics"]:
                if "agent_traits" not in subtopic:
                    subtopic["agent_traits"] = {}
        return jsonify(data)

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API key not found in .env")

    discussion_manager = DiscussionManager(api_key)
    app.config['discussion_manager'] = discussion_manager

    discussion_thread = threading.Thread(
        target=lambda: asyncio.run(discussion_manager.start_main_discussion()),
        daemon=True
    )
    discussion_thread.start()
    
    socketio.run(app, 
        host="0.0.0.0", 
        port=5000,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )