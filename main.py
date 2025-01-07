from openai import OpenAI
import asyncio, json, time, uuid, logging, os, threading, random
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
    ping_timeout=60,
    ping_interval=25
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

    async def start_main_discussion(self):
        topics = os.getenv("TOPICS").split(",")
        for topic in topics:
            main_topic = topic.strip()
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

    async def _generate_custom_agents(self, subtopic: str) -> List[dict]:
        prompt = f"""
        Create 4-6 specialized agents for discussing: {subtopic}
        Each agent should have unique expertise and a distinct personality profile.

        Return JSON:
        {{
          "agents": [
            {{
              "id": "Ob-[3 digit number]",
              "expertise": ["area1", "area2", "area3"],
              "personality": {{
                "core_trait": "primary personality characteristic",
                "traits": ["trait1", "trait2", "trait3"],
                "communication_style": "how they typically express themselves",
                "biases": ["potential biases they might have"],
                "strengths": ["key strengths"],
                "approach": "their general approach to problems/discussions"
              }},
              "system_prompt": "You are [ID], an expert in [expertise]. Your personality is [core_trait], you communicate [communication_style], and approach discussions by [approach]. [Rest of role and expertise context]"
            }}
          ]
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Design expert agents with rich personality profiles using Ob-[number] format IDs."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        try:
            result = json.loads(response.choices[0].message.content)
            agents = result["agents"]
            logging.info(f"Generated custom agents for '{subtopic}': {json.dumps(agents, indent=2)}")
            
            for agent in agents:
                agent["selected_personality"] = random.choice(agent["personality"]["traits"])
                
            return agents
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Error generating agents: {e}")
            return [{
                "id": "Ob-000", 
                "expertise": [subtopic],
                "personality": {
                    "core_trait": "analytical",
                    "traits": ["logical", "methodical"],
                    "communication_style": "direct and factual",
                    "biases": ["favors data-driven approaches"],
                    "strengths": ["systematic analysis"],
                    "approach": "breaking down complex topics systematically"
                },
                "system_prompt": "You are Ob-000, a general expert.",
                "selected_personality": "analytical"
            }]

    async def _get_next_subtopic(self) -> str:
        prompt = f"""
        Main topic: {self.discussion.main_topic}
        Previous subtopics: {', '.join(s.topic for s in self.discussion.subtopics)}
        Generate a specific, focused subtopic exploring a key aspect of the main topic.
        Respond with only the subtopic name.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a discussion coordinator responsible for guiding the conversation flow."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.7
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
        for _ in range(2):  # Number of discussion rounds
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
                await asyncio.sleep(1)  # Add delay between messages

    async def _generate_response(self, agent_info: dict, subtopic: Subtopic) -> str:
        history = "\n".join([f"{m.agent}: {m.content}" for m in subtopic.messages[-5:]])
        personality = agent_info["personality"]
        
        prompt = f"""
        {agent_info['system_prompt']}
        Personality Profile:
        - Core trait: {personality['core_trait']}
        - Communication style: {personality['communication_style']}
        - Current trait focus: {agent_info['selected_personality']}
        - Approach: {personality['approach']}

        Topic: {subtopic.topic}
        Previous messages:
        {history}
        
        Respond according to your expertise and personality profile.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Engage in focused discussion while maintaining your defined personality."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "Error generating response."

    async def _run_head_agent(self, subtopic: Subtopic):
        messages = "\n".join([f"{m.agent}: {m.content}" for m in subtopic.messages])
        prompt = f"""
        Analyze the discussion about "{subtopic.topic}" and provide:
        1. Key insights summary
        2. Action items
        Format: Summary: [insights] Action Items: [steps]
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a discussion coordinator summarizing key points and action items."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
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

    discussion_thread = threading.Thread(
        target=lambda: asyncio.run(DiscussionManager(api_key).start_main_discussion()),
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