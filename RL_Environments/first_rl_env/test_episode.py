"""
Evaluation script - runs a full RL episode via WebSocket.

WHY WEBSOCKET: The OpenEnv HTTP server creates a NEW environment
instance for each REST call (/reset, /step). State is NOT shared
between REST calls, so search results are lost before filter runs.
WebSocket maintains a persistent session where state persists.

Usage:
    1. Start container:
       docker run -p 8000:8000 -e RESEARCH_TASK=single_topic_retrieval research_env_env:latest
    2. pip install websockets
    3. python3 test_episode.py
"""

import json
import asyncio
import websockets


async def run_episode():
    uri = "ws://localhost:8000/ws"
    print("Connecting to", uri)

    async with websockets.connect(uri) as ws:
        # 1. RESET
        await ws.send(json.dumps({"type": "reset"}))
        resp = json.loads(await ws.recv())
        obs = resp.get("observation", resp)
        print("[DEBUG] Raw response:", json.dumps(resp, indent=2, default=str)[:800])
        print("=" * 60)
        print(f"Task: {obs.get('query', 'N/A')}")
        print(f"Phase: {obs.get('current_phase', 'N/A')}")
        print(f"Available: {obs.get('available_actions', 'N/A')}")
        # 2. SEARCH
        await ws.send(json.dumps({
            "type": "step",
            "action": {
                "action_type": "search",
                "query_terms": "transformer attention mechanism NLP"
            }
        }))
        resp = json.loads(await ws.recv())
        obs = resp.get("observation", resp)
        reward = resp.get("reward", obs.get("reward", 0))
        papers = obs.get("retrieved_papers", [])
        print(f"\nSEARCH: {len(papers)} papers | Reward: {reward:.4f}")
        for p in papers[:5]:
            print(f"  {p['paper_id']}: {p['title'][:55]}")
        if len(papers) > 5:
            print(f"  ...and {len(papers)-5} more")

        ids = [p["paper_id"] for p in papers]

        # 3. FILTER top 3
        fids = ids[:3] if len(ids) >= 3 else ids
        await ws.send(json.dumps({
            "type": "step",
            "action": {"action_type": "filter", "paper_ids": fids}
        }))
        resp = json.loads(await ws.recv())
        obs = resp.get("observation", resp)
        reward = resp.get("reward", obs.get("reward", 0))
        filtered = obs.get("filtered_papers", [])
        print(f"\nFILTER: {len(filtered)} kept | Reward: {reward:.4f}")
        print(f"  {obs.get('last_action_feedback','')}")

        # 4. SUMMARIZE first paper
        sid = filtered[0]["paper_id"] if filtered else (fids[0] if fids else "paper_001")
        await ws.send(json.dumps({
            "type": "step",
            "action": {
                "action_type": "summarize",
                "paper_id": sid,
                "content": (
                    "The Transformer introduces self-attention and multi-head "
                    "attention replacing recurrence with parallelization. It uses "
                    "positional encoding for sequence order and encoder-decoder "
                    "architecture for machine translation."
                )
            }
        }))
        resp = json.loads(await ws.recv())
        obs = resp.get("observation", resp)
        reward = resp.get("reward", obs.get("reward", 0))
        print(f"\nSUMMARIZE {sid} | Reward: {reward:.4f}")
        print(f"  {obs.get('last_action_feedback','')}")

        # 5. EXPLAIN
        await ws.send(json.dumps({
            "type": "step",
            "action": {
                "action_type": "explain",
                "paper_id": sid,
                "content": (
                    "Instead of reading words one by one, the Transformer looks "
                    "at all words at once using attention. This makes training "
                    "much faster. The model learns which words matter to each "
                    "other, like focusing on key words in a sentence."
                )
            }
        }))
        resp = json.loads(await ws.recv())
        obs = resp.get("observation", resp)
        reward = resp.get("reward", obs.get("reward", 0))
        print(f"\nEXPLAIN {sid} | Reward: {reward:.4f}")
        print(f"  {obs.get('last_action_feedback','')}")

        # 6. FINALIZE
        await ws.send(json.dumps({
            "type": "step",
            "action": {"action_type": "finalize"}
        }))
        resp = json.loads(await ws.recv())
        obs = resp.get("observation", resp)
        reward = resp.get("reward", obs.get("reward", 0))
        done = resp.get("done", obs.get("done", False))

        print("\n" + "=" * 60)
        print(f"FINAL SCORE: {reward:.4f}  Done: {done}")
        print(f"  {obs.get('last_action_feedback','')}")
        meta = obs.get("metadata", {})
        if meta:
            print(f"  Cumulative: {meta.get('cumulative_reward','N/A')}")
            print(f"  Task: {meta.get('task_name','N/A')} ({meta.get('difficulty','N/A')})")
            print(f"  History: {meta.get('action_history',[])}")
        print("=" * 60)

        if reward >= 0.8: print("EXCELLENT")
        elif reward >= 0.6: print("GOOD")
        elif reward >= 0.4: print("FAIR")
        elif reward >= 0.2: print("POOR")
        else: print("FAILING")


if __name__ == "__main__":
    asyncio.run(run_episode())
