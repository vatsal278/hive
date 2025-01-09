from openai import OpenAI
import asyncio, json, time, uuid, logging, os, threading
from pathlib import Path
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
from dotenv import load_dotenv
from dataclasses import dataclass, asdict, field
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, 
    cors_allowed_origins="*",
    async_mode="threading",
    logger=True,
    engineio_logger=True,
    allow_unsafe_werkzeug=True
)

def safe_emit(event, data):
    try:
        socketio.emit(event, data)
        logging.info(f"Emitted {event}: {data}")
    except Exception as e:
        logging.error(f"Socket emission failed: {e}")

@socketio.on('connect')
def connect():
    logging.info('Client connected')

@socketio.on('disconnect')
def disconnect():
    logging.info('Client disconnected')

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

class DiscussionManager:
    def __init__(self, api_key: str, storage_path: str = "discussions"):
        self.client = OpenAI(api_key=api_key)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.discussion = None
        self.current_topic_index = 0
        self.is_processing = False
        self.topics = []
        self.load_topics_from_env()

    def load_topics_from_env(self):
        """Load topics from environment variables."""
        load_dotenv(override=True)  # Reload .env file
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
                await asyncio.sleep(10)  # Wait for topics to be added
                continue

            if self.current_topic_index >= len(self.topics):
                self.current_topic_index = 0  # Reset to start if we've processed all topics
                await asyncio.sleep(10)  # Wait before starting new cycle
                self.load_topics_from_env()  # Check for new topics before starting new cycle
                continue

            main_topic = self.topics[self.current_topic_index]
            if not self.is_processing:
                self.is_processing = True
                logging.info(f"Starting discussion on: {main_topic}")
                self.discussion = Discussion(main_topic=main_topic, subtopics=[], current_subtopic_id=None)
                
                start_time = time.time()
                time_limit = 7200  # 2 hours in seconds
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
                self.load_topics_from_env()  # Check for new topics after completing current topic

    async def _generate_custom_agents(self, subtopic: str) -> List[dict]:
        prompt = f"""
        You are tasked with creating specialized agents for discussing: {subtopic}
        Each agent must:
        1. Have a unique ID:
           - Example: "Ob-123"
        2. Possess expertise in 2-3 specific areas related to the subtopic:
           - Example: ["cryptocurrency", "blockchain", "tokenomics"]
        3. Exhibit a unique personality with the following attributes:
           - Core Trait: Example: "analytical" or "empathetic"
           - Secondary Traits: Example: ["logical", "methodical", "creative"]
           - Communication Style: Example: "formal and structured" or "humorous and engaging"
           - Strengths: Example: ["ability to break down complex topics", "excellent at persuasion"]
           - Biases: Example: ["favors quantitative data", "prone to overestimating risks"]

        Return the agents in this exact JSON format:
        {{"agents":[{{"id":"Ob-123","expertise":["area1","area2"],"personality":{{"core_trait":"trait","traits":["trait1","trait2"],"communication_style":"style","strengths":["strength1"],"biases":["bias1"]}},"system_prompt":"You are Ob-123, expert in area1, area2..."}}]}}

        Return exactly 4 agents.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a JSON generator that creates agent profiles. Only return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            agents = result.get("agents", [])
            if not agents:
                raise ValueError("No agents in response")
                
            logging.info(f"Successfully generated {len(agents)} agents for '{subtopic}'")
            return agents
            
        except Exception as e:
            logging.error(f"Error generating agents: {e}")
            return [{
                "id": "Ob-000", 
                "expertise": [subtopic],
                "personality": {
                    "core_trait": "analytical",
                    "traits": ["logical", "methodical"],
                    "communication_style": "direct and factual",
                    "biases": ["favors data-driven approaches"],
                    "strengths": ["systematic analysis"]
                },
                "system_prompt": f"You are Ob-000, a general expert in {subtopic}."
            }]

    async def _get_next_subtopic(self) -> str:
        prompt = f"""
        Main topic: "{self.discussion.main_topic}"
        Previous subtopics: "{', '.join(s.topic for s in self.discussion.subtopics)}"
        Suggest the next subtopic to explore. Return only the subtopic name.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
You are a discussion coordinator responsible for guiding conversations.
Your task is to:
1. Suggest the next subtopic for a focused discussion.
2. Ensure the subtopic is specific, relevant to the main topic, and not redundant with previous subtopics.
3. Keep the subtopic concise (1 sentence or phrase).
Examples:
- Main topic: "Artificial Intelligence"
  Previous subtopics: ["AI in healthcare", "Ethics in AI"]
  Next subtopic: "AI in autonomous vehicles"
- Main topic: "Blockchain Technology"
  Previous subtopics: ["Smart contracts", "DeFi applications"]
  Next subtopic: "Blockchain scalability solutions"
Always return only the subtopic name without additional explanations.
"""},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.7,
            stop=["\n", ".", "Next:", "Topic:"]
        )
        return response.choices[0].message.content.strip()

    async def _run_discussion(self):
        subtopic = await self._get_next_subtopic()
        custom_agents = await self._generate_custom_agents(subtopic)

        agent_traits = {
            agent["id"]: {
                "core_trait": agent["personality"]["core_trait"],
                "traits": agent["personality"]["traits"],
                "expertise": agent["expertise"]
            } for agent in custom_agents
        }
        
        new_subtopic = Subtopic(
            id=str(uuid.uuid4()),
            topic=subtopic,
            agents=[agent["id"] for agent in custom_agents],
            messages=[],
            summary=None,
            start_time=time.time(),
            end_time=None,
            agent_traits=agent_traits
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

    async def _run_agent_discussion(self, subtopic: Subtopic, custom_agents: List[dict]):
        for _ in range(10):  # Number of discussion rounds
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
                await asyncio.sleep(3)  # Add delay between messages

    async def _generate_response(self, agent_info: dict, subtopic: Subtopic) -> str:
        recent_messages = subtopic.messages[-10:]
        
        messages = [
            {"role": "system", "content": """
            You are an AI agent participating in a collaborative and intense brainstorming session with other AI agents. 
            The discussion is focused on a specific subtopic derived from a larger main topic. Your task is to:
            1. Respond directly and concisely to the subtopic or recent conversation.
            2. Debate, brainstorm, and critically analyze ideas presented by other agents. Always aim to:
               - Challenge assumptions.
               - Identify problems and solutions.
               - Propose actionable insights or innovative ideas.
            3. Reference context from:
               - The immediate conversation (recent messages).
               - Medium-term context (earlier exchanges in this session).
               - Long-term context (the overarching subtopic and main topic).
            ### **Rules of Engagement**:
            - Do NOT greet other agents or introduce yourself. Assume all agents already know your expertise and personality.
            - Do NOT use polite or formal phrases like "I believe," "In my opinion," or "Thank you."
            - Assume all agents are experts in their fields, so avoid redundant explanations of basic concepts.
            - Assume you and the other agents share access to all historical messages and know the subtopic.
            ### **Examples of Engagement**:
            1. Subtopic: "Blockchain scalability solutions"
               - Recent Messages:
                 - Agent-1: "Layer-2 solutions like rollups can reduce costs."
                 - Agent-2: "Energy consumption for base-layer blockchains is a bottleneck."
               - Response:
                 - "Rollups are promising, but the interoperability between Layer-2 and the base layer is critical for adoption. Also, energy-efficient consensus mechanisms like Proof-of-Stake must scale alongside these solutions."
            2. Subtopic: "AI in autonomous vehicles"
               - Recent Messages:
                 - Agent-1: "AI models need real-time processing for decision-making."
                 - Agent-2: "Sensor fusion is an area that requires more research."
               - Response:
                 - "While real-time processing is essential, edge computing could address latency concerns. As for sensor fusion, integrating LiDAR and computer vision may solve the perception gap, but it increases hardware costs. Let's discuss how to optimize these trade-offs."
            3. Subtopic: "Ethics in AI"
               - Recent Messages:
                 - Agent-1: "Bias in training data remains a core ethical concern."
                 - Agent-2: "Transparency in AI decision-making is key."
               - Response:
                 - "Bias is fundamental, but let's focus on solutions like federated learning to diversify training data. Regarding transparency, interpretability tools are useful, but regulatory frameworks are also critical. Can we identify gaps in existing frameworks?"
            Your goal is to build upon these examples. Always engage critically, reference context, and propose actionable ideas. Avoid unnecessary elaboration or restating points already made.
            """}
        ]
    
        messages.extend([
            {"role": "assistant", "content": f"{msg.agent}: {msg.content}"}
            for msg in recent_messages
        ])

        messages.append({
            "role": "user",
            "content": f"""
            Subtopic: "{subtopic.topic}"
            Critically respond to the discussion so far, building on prior messages. 
            Respond with a single clear point or counterpoint in 4-5 sentences maximum.
            Stay focused on the subtopic and propose new ideas, counterpoints, or solutions.
            """
        })

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=250,
                temperature=0.6,
                top_p=0.7,
                frequency_penalty=1,
                presence_penalty=2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "Error generating response."

    async def _run_head_agent(self, subtopic: Subtopic):
        messages = "\n".join([f"{m.agent}: {m.content}" for m in subtopic.messages])
        prompt = f"""
        Analyze and summarize the discussion about "{subtopic.topic}" in chronological order.

        1. Participant Analysis:
           - List each participating agent's ID and their key areas of expertise
           - Note their unique perspectives and communication styles
           - Highlight how their specialized knowledge contributed to the discussion

        2. Discussion Overview:
           - Main focus and scope
           - Key themes that emerged
           - Overall direction of the conversation

        3. Chronological Development:
           - How the discussion evolved
           - Major turning points
           - Key transitions in topics

        4. Notable Contributions:
           - Significant insights by specific agents (include agent IDs)
           - Innovative ideas proposed
           - Important counterpoints raised
           - Breakthrough moments

        5. Key Conclusions:
           - Major points of consensus
           - Unresolved debates
           - Novel perspectives gained

        6. Action Items and Next Steps:
           - Concrete recommendations
           - Areas identified for further exploration
           - Practical applications suggested

        7. Recommended Next Topics:
           - Based on this discussion's outcomes
           - Consider unresolved points that need deeper exploration
           - Identify emerging themes that warrant dedicated discussion
           - Account for previous subtopics: {', '.join(s.topic for s in self.discussion.subtopics)}
           - Suggest 2-3 most promising directions for future discussions

        Format the summary with clear headings and ensure you capture the essence of how ideas built upon each other throughout the discussion.

        Previous discussion messages for reference:
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

    def _save_discussion(self):
        discussion_data = asdict(self.discussion)
        with open(self.storage_path / "discussion.json", "w") as f:
            json.dump(discussion_data, f, indent=2)

    async def _finalize_discussion(self):
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

@app.route("/discussions")
def get_discussions():
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