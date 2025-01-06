from openai import OpenAI
import asyncio
import nest_asyncio
from flask import Flask, jsonify
from flask_socketio import SocketIO
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import time, random
import uuid
import logging
from flask_cors import CORS
import threading, os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
nest_asyncio.apply()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", logger=True, engineio_logger=True)

@dataclass
class Message:
    agent: str
    content: str
    timestamp: float
    type: str  # 'chat', 'summary', 'selection'

@dataclass
class Subtopic:
    id: str
    topic: str
    agents: List[str]
    messages: List[Message]
    summary: Optional[str]
    start_time: float
    end_time: Optional[float]

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
        self.agent_personalities = [
            ("Ob-1", "AI and Data Strategist", 
             "You are Ob-1, an advanced AI and data expert. Share concrete insights about AI implementations, data architecture, and technological innovations."),
            
            ("Ob-2", "Decentralized Systems Specialist", 
             "You are Ob-2, a blockchain and Web3 strategist. Provide specific insights about blockchain technologies, protocols, and decentralized applications."),
            
            ("Ob-3", "Financial Architect", 
             "You are Ob-3, a finance and investment expert. Share specific market insights, investment strategies, and financial analysis."),
            
            ("Ob-4", "Biotech and Chemical Analyst", 
             "You are Ob-4, a specialist in biology and chemistry. Discuss practical applications of biotech, chemical processes, and scientific innovations."),
            
            ("Ob-5", "Theoretical Physicist and Mathematician", 
             "You are Ob-5, an expert in physics and mathematics. Explain complex systems through mathematical frameworks and physical principles."),
            
            ("Ob-6", "Cosmic Analyst and Historian", 
             "You are Ob-6, a cosmic strategist with expertise in astronomy and historical patterns. Connect celestial phenomena with practical implications."),
            
            ("Ob-7", "Behavioral and Cognitive Specialist", 
             "You are Ob-7, a psychologist and behavior analyst. Share insights about human behavior, decision-making, and cognitive optimization."),
            
            ("Ob-8", "Environmental and Sustainability Strategist", 
             "You are Ob-8, an environmental expert. Provide practical sustainability solutions and environmental impact analysis.")
        ]
        
        self.head_agent = ("HeadOp", "Strategic Coordinator", 
                          "You are HeadOp. Generate focused, relevant subtopics that directly explore key aspects of the main topic. For example, if discussing Cryptocurrency, generate subtopics like 'DeFi Protocols', 'NFT Marketplaces', 'Layer 2 Scaling Solutions', etc.")

    async def start_discussion(self, main_topic: str):
        self.discussion = Discussion(main_topic=main_topic, subtopics=[], current_subtopic_id=None)
        await self._run_discussion()

    def _construct_prompt(self, agent_name: str, subtopic: Subtopic, history: List[dict]) -> str:
        history_text = "\n".join(f"{m['agent']}: {m['content']}" for m in history[-5:])
        personality = self._get_agent_personality(agent_name)

        return f"""
        {personality}

        Current Discussion:
        {history_text}

        Communication Guidelines:
        - Share direct insights and build on others' points
        - Challenge or support ideas with specific examples
        - Ask questions to other agents when relevant
        - Keep the conversation flowing naturally
        - Focus on practical, implementable ideas
        """

    async def _run_discussion(self):
        while True:
            subtopic = await self._get_next_subtopic()
            selected_agents = random.sample([agent[0] for agent in self.agent_personalities], 5)

            new_subtopic = Subtopic(
                id=str(uuid.uuid4()),
                topic=subtopic,
                agents=selected_agents,
                messages=[],
                summary=None,
                start_time=time.time(),
                end_time=None
            )

            self.discussion.subtopics.append(new_subtopic)
            self.discussion.current_subtopic_id = new_subtopic.id
            socketio.start_background_task(socketio.emit, "current_topic_stream", {"subtopic": new_subtopic.topic})

            await self._run_agent_discussion(new_subtopic)
            await self._run_head_agent(new_subtopic)
            new_subtopic.end_time = time.time()
            self._save_discussion()

            self.discussion.current_subtopic_id = None
            await asyncio.sleep(5)

    async def _get_next_subtopic(self) -> str:
        prompt = f"""
        Main topic: {self.discussion.main_topic}
        
        Previous subtopics: {', '.join(s.topic for s in self.discussion.subtopics)}
        
        Generate a specific, focused subtopic that explores a key aspect of {self.discussion.main_topic}. 
        The subtopic should be concrete and directly related to the main topic.
        
        Example format for different domains:
        - Crypto: "DeFi Lending Protocols", "NFT Gaming Platforms", "Cross-chain Bridges"
        - AI: "Large Language Models", "Computer Vision Applications", "Reinforcement Learning"
        - Space: "Mars Habitat Design", "Resource Extraction Methods", "Life Support Systems"
        
        Respond with only the subtopic name, no additional text.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate focused, specific subtopics that directly relate to the main topic."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    async def _run_agent_discussion(self, subtopic: Subtopic):
        for _ in range(8):  # Reduced number of turns for more focused discussion
            for agent in subtopic.agents:
                prompt = self._construct_prompt(agent, subtopic, [asdict(m) for m in subtopic.messages])
                response = await self._generate_response(agent, prompt)

                message = Message(agent=agent, content=response, timestamp=time.time(), type="chat")
                subtopic.messages.append(message)
                socketio.start_background_task(socketio.emit, "current_topic_stream", {"message": asdict(message)})
                await asyncio.sleep(2)

    async def _generate_response(self, agent_name: str, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Engage in natural discussion with specific, meaningful contributions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3,
                presence_penalty=0.8,
                frequency_penalty=0.9
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error generating response for {agent_name}: {e}")
            return "Error generating response."

    async def _run_head_agent(self, subtopic: Subtopic):
        messages = "\n".join([f"{m.agent}: {m.content}" for m in subtopic.messages])
        
        prompt = f"""
        Analyze the discussion about "{subtopic.topic}" and provide:
        1. A concise summary of key insights and conclusions
        2. Important action items or next steps discussed
        
        Format your response as:
        Summary: [1-2 sentences of key insights]
        Action Items: [2-3 concrete next steps]
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.head_agent[2]},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        summary = response.choices[0].message.content.strip()
        subtopic.summary = summary
        socketio.start_background_task(socketio.emit, "current_topic_stream", {"summary": summary})

    def _get_agent_role(self, agent_name: str) -> str:
        for agent, role, _ in self.agent_personalities:
            if agent == agent_name:
                return role
        return "Unknown Role"

    def _get_agent_personality(self, agent_name: str) -> str:
        for agent, _, personality in self.agent_personalities:
            if agent == agent_name:
                return personality
        return "Unknown personality"

    def _save_discussion(self):
        file_path = self.storage_path / "discussion.json"
        with open(file_path, "w") as f:
            json.dump(asdict(self.discussion), f, indent=2)

@app.route("/discussions")
def get_discussions():
    file_path = Path("discussions/discussion.json")
    if not file_path.exists():
        return jsonify([])
    with open(file_path, "r") as f:
        discussion = json.load(f)
    return jsonify([{"id": s["id"], "topic": s["topic"], "agents": s["agents"], "summary": s["summary"]} 
                   for s in discussion["subtopics"]])

@app.route("/discussions/<subtopic_id>")
def get_subtopic(subtopic_id):
    file_path = Path("discussions/discussion.json")
    if not file_path.exists():
        return jsonify({"error": "Discussion not found"}), 404
    with open(file_path, "r") as f:
        discussion = json.load(f)
        for subtopic in discussion["subtopics"]:
            if subtopic["id"] == subtopic_id:
                return jsonify(subtopic)
    return jsonify({"error": "Subtopic not found"}), 404

def start_discussion(topic: str, api_key: str):
    manager = DiscussionManager(api_key=api_key)
    asyncio.run(manager.start_discussion(topic))

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("API_KEY")

    if not api_key:
        raise ValueError("OpenAI API key is not set in the .env file.")

    discussion_thread = threading.Thread(
        target=start_discussion,
        args=("Cryptocurrency Evolution and Future", api_key),
        daemon=True,
    )

    discussion_thread.start()
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)