## Connect to this environment

#### Connect from Python using ResearchEnv:

```
from research_assistant_env import ResearchAction, ResearchEnv

with ResearchEnv.from_env("noobcoder27/research_assistant_env") as env:
    result = await env.step(ResearchAction(message="..."))
```

#### Or connect directly to a running server:
```env = ResearchEnv(base_url="http://localhost:8000")```

## *Contribute to this environment*
#### Submit improvements via pull request on the Hugging Face Hub.
```
openenv fork noobcoder27/research_assistant_env --repo-id <your-username>/<your-repo-name>
```

#### Then make your changes and submit a pull request:
```
cd <forked-repo>
openenv push noobcoder27/research_assistant_env --create-pr
```

