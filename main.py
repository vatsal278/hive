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

# Logging Configuration
logging.basicConfig(level=logging.INFO)
nest_asyncio.apply()

# Flask and SocketIO Setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Dataclasses for Discussion Structure
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

# Discussion Manager
class DiscussionManager:
    def __init__(self, api_key: str, storage_path: str = "discussions"):
        self.client = OpenAI(api_key=api_key)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.discussion = None
        self.agent_personalities = [
            ("Agent Neuralis", "Data, AI, and Coding Specialist", "You are Agent Neuralis, an expert in AI and coding."),
            ("Agent Cryptonova", "Crypto and Decentralized Systems Specialist", "You are Agent Cryptonova, an expert in crypto and Web3."),
            ("Agent Money", "Capitalist and Finance Expert", "You are Agent Money, a financial expert."),
            ("Agent BioSparkle", "Biologist and Chemistry Expert", "You are Agent BioSparkle, an expert in biology."),
            ("Agent Infinity", "Physics and Mathematics Specialist", "You are Agent Infinity, an expert in physics."),
            ("Agent Celestia", "Astrology and Ancient Knowledge Expert", "You are Agent Celestia, an expert in astrology."),
            ("Agent PsyLin", "Psychologist and Behavior Expert", "You are Agent PsyLin, an expert in psychology."),
            ("Agent Earth", "Environment and Sustainability Expert", "You are Agent Earth, an expert in sustainability."),
        ]

    async def start_discussion(self, main_topic: str):
        self.discussion = Discussion(main_topic=main_topic, subtopics=[], current_subtopic_id=None)
        await self._run_discussion()

    def _construct_prompt(self, agent_name: str, subtopic: Subtopic, history: List[dict]) -> str:
        history_text = "\n".join(f"{m['agent']}: {m['content']}" for m in history[-5:])
        participants = ", ".join([f"{agent} ({self._get_agent_role(agent)})" for agent in subtopic.agents])
        personality = self._get_agent_personality(agent_name)
        return f"""
        Main Topic: {self.discussion.main_topic}
        Subtopic: {subtopic.topic}
        Participants: {participants}

        Previous Messages:
        {history_text}

        {personality}
        """

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
                end_time=None,
            )

            self.discussion.subtopics.append(new_subtopic)
            self.discussion.current_subtopic_id = new_subtopic.id
            self._save_discussion()

            await self._run_agent_discussion(new_subtopic)
            summary = await self._generate_summary(new_subtopic)
            new_subtopic.summary = summary
            new_subtopic.end_time = time.time()
            self._save_discussion()
            await asyncio.sleep(10)

    async def _get_next_subtopic(self) -> str:
        return "AI System Architecture"  # Mocked subtopic for simplicity

    async def _run_agent_discussion(self, subtopic: Subtopic):
        for _ in range(10):
            for agent in subtopic.agents:
                prompt = self._construct_prompt(agent, subtopic, [asdict(m) for m in subtopic.messages])
                response = await self._generate_response(agent, prompt)

                message = Message(agent=agent, content=response, timestamp=time.time(), type="chat")
                subtopic.messages.append(message)
                self._save_discussion()

                if subtopic.id == self.discussion.current_subtopic_id:
                    socketio.emit("message_stream", asdict(message))
                await asyncio.sleep(2)

    async def _generate_response(self, agent_name: str, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are {agent_name}."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.7,
                stop=["\n\n"],
            )
            
            # Log the raw response
            logging.info(f"Raw response for {agent_name}: {response}")
            
            # Extract the content from the response
            content = response.choices[0].message.content.strip()
            logging.info(f"Parsed response for {agent_name}: {content}")
            return content
        except Exception as e:
            logging.error(f"Error generating response for {agent_name}: {e}")
            return "Error generating response."

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
    return jsonify([{"id": s["id"], "topic": s["topic"], "agents": s["agents"], "summary": s["summary"]} for s in discussion["subtopics"]])

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

# Main Execution
if __name__ == "__main__":
    # Load environment variables from the .env file
    load_dotenv()
    # Access the API key
    api_key = os.getenv("API_KEY")

    # Verify the API key is loaded
    if not api_key:
        raise ValueError("OpenAI API key is not set in the .env file.")

    discussion_thread = threading.Thread(
        target=start_discussion,
        args=("AI Ethics and Development", ""),
        daemon=True,
    )

    discussion_thread.start()

    socketio.run(app, port=5000)

