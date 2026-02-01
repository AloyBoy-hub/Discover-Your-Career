import os
import json
import requests
from openai import OpenAI

# Base URL for roadmap.sh raw data
GITHUB_BASE = "https://raw.githubusercontent.com/kamranahmedse/developer-roadmap/master/src/data/roadmaps"

# Helper to get client
from typing import Optional

# Mapping of common roles to the ID used in the repo
ROLE_MAPPING = {
    "frontend": "frontend",
    "frontend developer": "frontend",
    "backend": "backend",
    "backend developer": "backend",
    "fullstack": "full-stack",
    "full stack developer": "full-stack",
    "devops": "devops",
    "android": "android",
    "ios": "ios",
    "data analyst": "data-analyst",
    "ai data scientist": "ai-data-scientist",
    "data scientist": "ai-data-scientist",
    "machine learning": "ai-data-scientist",
    "ux design": "ux-design",
    "react": "react" 
}

def fetch_roadmap_json(role_slug):
    """
    Fetches the raw JSON content from roadmap.sh GitHub repo.
    """
    # Try role.json
    url = f"{GITHUB_BASE}/{role_slug}/{role_slug}.json"
    print(f"Fetching roadmap from: {url}")
    
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"Failed to fetch {url}: {resp.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching raw roadmap: {e}")
        return None

def generate_roadmap(current_skills: list[str], target_role: str, client: Optional[OpenAI] = None):
    """
    1. Fetches official roadmap JSON.
    2. Uses LLM to check which topics user already knows.
    3. Annotates the JSON nodes with 'status': 'completed' | 'pending'.
    """
    # 1. Resolve API Key and Base URL
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "API Key (DEEPSEEK_API_KEY or OPENAI_API_KEY) not set"}

    # Use provided client or create one based on available key
    if client is None:
        base_url = "https://api.deepseek.com" if os.environ.get("DEEPSEEK_API_KEY") else None
        client = OpenAI(api_key=api_key, base_url=base_url)

    # 1. Resolve Role ID
    role_slug = ROLE_MAPPING.get(target_role.lower())
    if not role_slug:
        # Fallback: try using the target_role as slug directly
        role_slug = target_role.lower().replace(" ", "-")

    roadmap_data = fetch_roadmap_json(role_slug)
    
    if not roadmap_data:
        return {
            "error": f"Could not find official roadmap for role: {target_role}. Please try 'frontend', 'backend', 'devops', etc."
        }

    # 2. Extract Topics
    # roadmap_data['nodes'] contains items with "type": "topic" and "data": {"label": "..."}
    nodes = roadmap_data.get('nodes', [])
    topic_nodes = [n for n in nodes if n.get('type') == 'topic']
    
    # Create a list of labels to send to LLM
    # We map Label -> Node ID to easily update later
    label_to_id = {n['data']['label']: n['id'] for n in topic_nodes}
    all_topics = list(label_to_id.keys())
    
    # If list is too huge, we might need to chunk, but typically roadmaps have <200 nodes.
    
    # Removed redundant client instantiation to avoid NameError and reuse previous setup
    
    prompt = f"""
    You are a Career Architect.
    
    User Skills: {json.dumps(current_skills)}
    Target Role: {target_role}
    Roadmap Topics: {json.dumps(all_topics)}
    
    Task 1: Identify "Known Topics" (fuzzy match User Skills to Roadmap Topics).
    Task 2: Select the most critical "Missing Topics" for this specific role.
    Task 3: Group these Missing Topics into a 3-part execution plan.
    
    Return JSON ONLY:
    {{
        "known_topics": ["Topic A", "Topic B"],
        "plan": {{
            "short_term": [
                {{"topic": "Critical Basic Skill", "reason": "Necessary for entry-level work", "estimated_hours": 10}}
            ],
            "medium_term": [
                {{"topic": "Core Advanced Skill", "reason": "Standard industry requirement", "estimated_hours": 20}}
            ],
            "long_term": [
                {{"topic": "Niche/Expert Skill", "reason": "For senior differentiation", "estimated_hours": 40}}
            ]
        }}
    }}
    """
    
    model_name = "deepseek-chat" if os.environ.get("DEEPSEEK_API_KEY") else "gpt-4o-mini"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful career architect that outputs strict JSON."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        
        known_topics = set(result.get('known_topics', []))
        personal_plan = result.get('plan', {})
        
        # 3. Annotate Nodes
        enriched_nodes = []
        missing_skills_list = []
        
        for node in nodes:
            # Default status
            node_status = "pending"
            
            if node.get('type') == 'topic':
                label = node.get('data', {}).get('label', '')
                if label in known_topics:
                    node_status = "completed"
                else:
                    missing_skills_list.append(label)
            
            # Inject status into data
            if 'data' not in node:
                node['data'] = {}
            node['data']['status'] = node_status
            
            enriched_nodes.append(node)
            
        # Reconstruct Graph
        roadmap_data['nodes'] = enriched_nodes
        
        return {
            "role": target_role,
            "roadmap_graph": roadmap_data, 
            "summary": {
                "total_topics": len(topic_nodes),
                "completed": len(known_topics),
                "progress": f"{len(known_topics)}/{len(topic_nodes)}"
            },
            "learning_plan": personal_plan 
        }

    except Exception as e:
        print(f"DeepSeek/Processing Error: {e}")
        return {"error": str(e)}
