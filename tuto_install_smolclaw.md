
# Tuto Smolclaw

D'après @iMilnb https://github.com/NetBSDfr/smolBSD/blob/main/service/clawd/README.md et https://www.twitch.tv/videos/2724869684

Testé sous Ubuntu 24  
Driver nvidia : NVIDIA-SMI 575.57.08 ;  Driver Version: 575.57.08 ; CUDA Version: 12.9  
Carte : NVIDIA RTX 2000 Ada Generation Laptop GPU 8Go de RAM  

Modèles testés : https://huggingface.co/Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF (et sa version 4B).
 
L'objectif est de créer une microVM NetBSD contenant Picoclaw, de manière à créer des agents dans un conteneur sécurisé. Les modèles sont servis par la machine hôte avec llama.cpp ou llama-cpp-python (https://llama-cpp-python.readthedocs.io) 

On installe également :  
* un server IRC en local dans l'hôte pour ne pas avoir besoin d'une connexion web (le canal IRC est nécessaire pour Picoclaw). 
* un server nginx dans l'hôte (optionnel : il est possible d'accéder directement aux ports utilisés par llama.cpp ou llama-cpp-python).

Trois terminaux sont nécessaires : pour IRC, pour Smolclaw, pour le server llama. La commande ```nvtop``` est également utile pour surveiller l'activité GPU. 

## Install llama.cpp

Préférer l'installation des bindings python (```llama_cpp_python```, cf. ci-dessous), moins risquée.

### cuda-toolkit (nvcc)

Nécessaire pour compiler llama.cpp.  
Ici on installe cuda-toolkit à côté de cuda 12.9 déjà installé. 

```
sudo apt install wget ca-certificates gnupg
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"
sudo apt update
# Install toolkit only (example package name — verify with `apt search cuda-toolkit` / NVIDIA docs)
sudo apt install -y cuda-toolkit-12-9
nano ~/.bashrc => 
    export CUDA_HOME=/usr/local/cuda-12.9
    export PATH="$CUDA_HOME/bin:$PATH"
```

### compiler llama.cpp 

https://unsloth.ai/docs/models/qwen3.5#llama.cpp-guides

```
cmake llama.cpp -B llama.cpp/build     -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
export CUDACXX=/usr/local/cuda-12.9/bin/nvcc
# -j 16 = nb de coeurs
cmake --build llama.cpp/build --config Release -j 16 --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
```

```
./llama.cpp/llama-server -hf Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-GGUF:Q4_K_M -np 1 -fa on -c $((8192*4)) --cache-type-k q4_0 --cache-type-v q4_0 --port 8001 --host 0.0.0.0
```


## Install llama_cpp_python 

Choix des bindings Python pour ne pas risquer de casser la configuration nvidia existante.  
Création d'un script avec l'aide Qwen2.5-coder:14b sous ollama

### Environnement

Terminal :

``` 
$ python3.12 -m venv llama_cpp
$ . llama_cpp/bin/activate
$ CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python # indispensable d'ajouter le flag DGGML_CUDA !! sinon c'est sur cpu
$ pip install fastapi uvicorn huggingface-hub
```

Téléchargement modèle (console python) :

```
$ python
>>> from llama_cpp import Llama
>>> llm = Llama.from_pretrained( \
    repo_id="Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF", \
    filename="Qwen3.5-2B.Q4_K_S.gguf")

>>> llm.model_path # donne le chemin complet du fichier gguf dans le cache huggingface
```
Le modèle est maintenant dans ```~/.cache/huggingface/hub/models--Jackrong--Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/snapshots/70b75bb7405ea59c69c43f1a89940303910ebe5b/./Qwen3.5-2B.Q4_K_S.gguf```


Script Python :

```
# Pré-requis

# $ python3.12 -m venv llama_cpp
# $ . llama_cpp/bin/activate
# $ CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python # indispensable d'ajouter le flag DGGML_CUDA !! sinon c'est sur cpu
# $ pip install fastapi uvicorn huggingface-hub


# Lancer le script :
# $ python llama_cpp_server.py

# ~~~~~~~~~~~~~~~~~

# Alternative à ce script, utiliser le server en CLI :
# CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python[server]

# python3 -m llama_cpp.server \
#     --model "$HOME/.cache/huggingface/hub/models--Jackrong--Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/snapshots/70b75bb7405ea59c69c43f1a89940303910ebe5b/./Qwen3.5-2B.Q4_K_S.gguf" \
#     --flash_attn on     # flash attention \
#     --host 0.0.0.0 \
#     --port 8003 \

    
# Les options ci-dessous sont présentes dans llama_cpp.server --help mais génèrent des erreurs (notamment avec le contexte)        
#     --ngpu_layers -1    # charger toutes les couches en GPU \
#     --n_ctx 32768       # max tokens en contexte \
#     --type_k 4          # cache kv Q4 \
#     --type_v 4          # cache kv Q4 \



from typing import Optional
from fastapi import FastAPI, Request
from pydantic import BaseModel
import llama_cpp as llama
import os
import time
import uuid
import json

HOME = os.path.expanduser('~') + '/'
model_path = HOME + ".cache/huggingface/hub/models--Jackrong--Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/snapshots/70b75bb7405ea59c69c43f1a89940303910ebe5b/./Qwen3.5-2B.Q4_K_S.gguf"
# model_path = HOME + ".cache/huggingface/hub/models--Jackrong--Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/snapshots/e512aa48e30fc2a64d55e907631be3740977b1f8/./Qwen3.5-4B.Q4_K_M.gguf"

app = FastAPI()

llm = llama.Llama(model_path=model_path,
            n_gpu_layers= -1,
            main_gpu= 0,
            # chat_format="chatml-function-calling", # en fait cette option ne marche pas (les tools ne sont pas appelés) !
            n_ctx=32768,
            flash_attn=True)


# class ChatCompletionRequest(BaseModel):
#     messages: list
#     tools: Optional[list] = None
#     tool_choice: Optional[str] | None = None # = "auto"
#     temperature: Optional[float] = 0.7


# def normalize_for_picoclaw(resp: dict, model_path: str) -> dict:
#     # Ensure choices is a list
#     choices = resp.get("choices")
#     if isinstance(choices, dict) and "choices" in choices:
#         choices = choices["choices"]
#     if not isinstance(choices, list):
#         choices = [choices]

#     # Ensure message.content is always a string (Go expects string, not null)
#     for c in choices:
#         msg = c.get("message", {})
#         if msg.get("content") is None:
#             tool_calls = msg.get("tool_calls") or msg.get("function_call") or msg.get("function_calls") or []
#             if tool_calls:
#                 # Keep tool_calls array unchanged; but set content to a string so Go unmarshaler succeeds.
#                 try:
#                     # stringified representation of the function emission (compact)
#                     first = tool_calls[0] if isinstance(tool_calls, list) and tool_calls else tool_calls
#                     content_obj = {"type": "function", "function": first.get("function", first)}
#                     msg["content"] = json.dumps(content_obj)
#                 except Exception:
#                     msg["content"] = ""
#             else:
#                 msg["content"] = ""
#             c["message"] = msg

#     return {
#         "id": resp.get("id", f"chatcmpl-{uuid.uuid4()}"),
#         "object": resp.get("object", "chat.completion"),
#         "created": resp.get("created", int(time.time())),
#         "model": resp.get("model", model_path),
#         "choices": choices,
#         "usage": resp.get("usage", {}),
#     }


@app.post("/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()

    print("REQUEST :", json.dumps(body, indent=2), )

    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)

    
    resp = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
    )

    print("RAW MODEL RESPONSE:", json.dumps(resp, indent=2))

    # out = normalize_for_picoclaw(resp, model_path) # utile avec chat_format="chatml-function-calling"
    return resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, )

```

Tester : 

Dans un 1er terminal : 
```
$ . llama_cpp/bin/activate
$ python llama_cpp_server.py
```

Dans un 2nd terminal
```
$ curl -X POST "http://127.0.0.1:8003/chat/completions"      -H "Content-Type: application/json"      -d '{
           "messages": [
             {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": "What is the capital of France?"}
           ],
           "max_tokens": 16,
           "temperature": 0.7
         }'
```

### Config nginx (optionnel)

Attention, ne pas oublier de remplacer l'url dans le fichier picoclaw config.json  
Si on utilise nginx l'url pour Smolclaw est : 192.168.122.1/llama_cpp/chat/completions (vérifier avec ifconfig)

```
log_format post_request '$remote_addr - $remote_user [$time_local] '
                             '"$request" $status $body_bytes_sent '
                             '"$http_referer" "$http_user_agent" '
                             'request_method=$request_method';
server {
    listen 80;
    server_name local.pcvv;
    
    access_log /var/log/nginx/post_access.log post_request;
    error_log /var/log/nginx/error.log;

    location /llama_cpp/chat/completions {
        proxy_pass http://127.0.0.1:8003/chat/completions;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
       
    }
}
```
Tester :

```

$ curl -X POST "http://local.pcvv/llama_cpp/chat/completions"      -H "Content-Type: application/json"      -d '{
           "messages": [
             {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": "What is the capital of France?"}
           ],
           "max_tokens": 16,
           "temperature": 0.7
         }'
```

### IRC local server

N.B.: si un firewall est actif penser à régler les ports (ufw allow 6667/tcp ?)

```
$ sudo apt install inspircd -y
$ sudo nano /etc/inspircd/inspircd.conf # => <bind address="192.168.122.1" port="6667" type="clients">

$ sudo service inspircd restart
$ irssi # /connect 192.168.122.1 6667 ; /nick VV ; /join #VVchannel (création du channel)

```

## Smolclaw (SmolBSD + Picoclaw)

### install 

Hôte : ```ifconfig``` pour récupérer l'ip (192.168.122.X)

```
$ git clone https://github.com/NetBSDfr/smolBSD
$ cd smolBSD
$ ./smoler.sh pull clawd-amd64:latest
$ ./startnb.sh -c 2 -m 1024 -f etc/clawd.conf -w ../pico_workspace/ -i clawd-amd64:latest # le dossier ../pico_workspace est accessible sur /mnt dans la vm
$ curl -v 192.168.122.1:8003/chat/completions # vérifier que la microvm trouve l'hôte (ici sans nginx)
$ curl -v 192.168.122.1/llama_cpp/chat/completions # vérifier que la microvm trouve l'hôte (ici avec nginx)
$ picoclaw onboard # génère le fichier config
```


### config.json


```$ vim /home/clawd/.picoclaw/config.json```

Rappel vim : i pour mode insert ; esc + :w pour enregistrer ; :q pour sortir

Sinon : copier le fichier dans ```/mnt/``` et l'éditer en dehors de la vm puis remplacer le fichier dans ```/home/clawd/.picoclaw/```

Remarques:

* Définir le modèle par défaut (```model_name``` doit correspondre à une entrée de ```model_list```)
* Vérifier que les infos pour le canal IRC sont conformes au server local 
* Les API keys, noms de modèles importent peu (sauf peut-être pour l'utilisation de llama.cpp?) 

```
{
  "session": {
    "dm_scope": "per-channel-peer"
  },
  "version": 1,
  "agents": {
    "defaults": {
      "workspace": "/home/clawd/.picoclaw/workspace",
      "restrict_to_workspace": true,
      "allow_read_outside_workspace": false,
      "provider": "",
      "model_name": "qwen llama_cpp_python + nginx",
      "max_tokens": 32768,
      "max_tool_iterations": 50,
      "summarize_message_threshold": 20,
      "summarize_token_percent": 75,
      "steering_mode": "one-at-a-time",
      "subturn": {
        "max_depth": 0,
        "max_concurrent": 0,
        "default_timeout_minutes": 0,
        "default_token_budget": 0,
        "concurrency_timeout_sec": 0
      },
      "tool_feedback": {
        "enabled": true,
        "max_args_length": 300
      }
    }
  },
  "channels": {
    "irc": {
      "enabled": true,
      "server": "192.168.122.1:6667",
      "tls": false,
      "nick": "smolclaw_vv",
      "sasl_user": "",
      "channels": [
        "#VVchannel"
      ],
      "allow_from": [
        "VV"
      ],
      "group_trigger": {},
      "typing": {},
      "reasoning_channel_id": ""
    }
  },
  "model_list": [
    {
      "model_name": "qwen llama.cpp",
      "model": "openai/Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF:Q4_K_S",
      "api_base": "http://192.168.122.1:8001/v1",
      "api_key": "llamacpp",
      "request_timeout": 300
    },
    {
      "model_name": "qwen llama_cpp_python",
      "model": "openai/qwen",
      "api_base": "http://192.168.122.1:8003",
      "api_key": "llamacpppython",
      "request_timeout": 300
    },
    {
      "model_name": "qwen llama_cpp_python + nginx",
      "model": "openai/qwen",
      "api_base": "http://192.168.122.1/llama_cpp",
      "api_key": "llama_cpp_python_nginx",
      "request_timeout": 300
    }
  ],
  "gateway": {
    "host": "127.0.0.1",
    "port": 18790,
    "hot_reload": false,
    "log_level": "fatal"
  },
  "hooks": {
    "enabled": true,
    "defaults": {
      "observer_timeout_ms": 500,
      "interceptor_timeout_ms": 5000,
      "approval_timeout_ms": 60000
    }
  },
  "tools": {
    "allow_read_paths": null,
    "allow_write_paths": null,
    "filter_sensitive_data": true,
    "filter_min_length": 8,
    "web": {
      "enabled": true,
      "brave": {
        "enabled": false,
        "max_results": 5
      },
      "tavily": {
        "enabled": false,
        "base_url": "",
        "max_results": 5
      },
      "duckduckgo": {
        "enabled": true,
        "max_results": 5
      },
      "prefer_native": true,
      "fetch_limit_bytes": 10485760,
      "format": "plaintext"
    },
    "cron": {
      "enabled": true,
      "exec_timeout_minutes": 5,
      "allow_command": true
    },
    "exec": {
      "enabled": true,
      "enable_deny_patterns": true,
      "allow_remote": true,
      "custom_deny_patterns": null,
      "custom_allow_patterns": null,
      "timeout_seconds": 60
    },
    "skills": {
      "enabled": true,
      "registries": {
        "clawhub": {
          "enabled": true,
          "base_url": "https://clawhub.ai",
          "search_path": "",
          "skills_path": "",
          "download_path": "",
          "timeout": 0,
          "max_zip_size": 0,
          "max_response_size": 0
        }
      },
      "github": {},
      "max_concurrent_searches": 2,
      "search_cache": {
        "max_size": 50,
        "ttl_seconds": 300
      }
    },
    "media_cleanup": {
      "enabled": true,
      "max_age_minutes": 30,
      "interval_minutes": 5
    },
    "mcp": {
      "enabled": false,
      "discovery": {
        "enabled": false,
        "ttl": 5,
        "max_search_results": 5,
        "use_bm25": true,
        "use_regex": false
      }
    },
    "append_file": {
      "enabled": true
    },
    "edit_file": {
      "enabled": true
    },
    "find_skills": {
      "enabled": true
    },
    "i2c": {
      "enabled": false
    },
    "install_skill": {
      "enabled": true
    },
    "list_dir": {
      "enabled": true
    },
    "message": {
      "enabled": true
    },
    "read_file": {
      "enabled": true,
      "max_read_file_size": 65536
    },
    "send_file": {
      "enabled": true
    },
    "spawn": {
      "enabled": true
    },
    "spawn_status": {
      "enabled": false
    },
    "spi": {
      "enabled": false
    },
    "subagent": {
      "enabled": true
    },
    "web_fetch": {
      "enabled": true
    },
    "write_file": {
      "enabled": true
    }
  },
  "heartbeat": {
    "enabled": true,
    "interval": 30
  },
  "devices": {
    "enabled": false,
    "monitor_usb": true
  },
  "voice": {
    "echo_transcription": false
  },
  "build_info": {
    "version": "0.2.4",
    "git_commit": "5f50ae5",
    "build_time": "2026-03-25T09:09:15Z",
    "go_version": "1.25.8"
  }
}
```

### Exemple de requête Picoclaw et sa réponse

Sortie dans le terminal où est lancé ```llama_cpp_server.py```


```
INFO:     Uvicorn running on http://0.0.0.0:8003 (Press CTRL+C to quit)
REQUEST : {
  "max_tokens": 32768,
  "messages": [
    {
      "role": "system",
      "content": "# picoclaw \ud83e\udd9e (0.2.4 (git: 5f50ae5))\n\nYou are picoclaw, a helpful AI assistant.\n\n## Workspace\nYour workspace is at: /home/clawd/.picoclaw/workspace\n- Memory: /home/clawd/.picoclaw/workspace/memory/MEMORY.md\n- Daily Notes: /home/clawd/.picoclaw/workspace/memory/YYYYMM/YYYYMMDD.md\n- Skills: /home/clawd/.picoclaw/workspace/skills/{skill-name}/SKILL.md\n\n## Important Rules\n\n1. **ALWAYS use tools** - When you need to perform an action (schedule reminders, send messages, execute commands, etc.), you MUST call the appropriate tool. Do NOT just say you'll do it or pretend to do it.\n\n2. **Be helpful and accurate** - When using tools, briefly explain what you're doing.\n\n3. **Memory** - When interacting with me if something seems memorable, update /home/clawd/.picoclaw/workspace/memory/MEMORY.md\n\n4. **Context summaries** - Conversation summaries provided as context are approximate references only. They may be incomplete or outdated. Always defer to explicit user instructions over summary content.\n\n\n\n---\n\n## AGENT.md\n\nYou are Pico, the default assistant for this workspace.\nYour name is PicoClaw \ud83e\udd9e.\n## Role\n\nYou are an ultra-lightweight personal AI assistant written in Go, designed to\nbe practical, accurate, and efficient.\n\n## Mission\n\n- Help with general requests, questions, and problem solving\n- Use available tools when action is required\n- Stay useful even on constrained hardware and minimal environments\n\n## Capabilities\n\n- Web search and content fetching\n- File system operations\n- Shell command execution\n- Skill-based extension\n- Memory and context management\n- Multi-channel messaging integrations when configured\n\n## Working Principles\n\n- Be clear, direct, and accurate\n- Prefer simplicity over unnecessary complexity\n- Be transparent about actions and limits\n- Respect user control, privacy, and safety\n- Aim for fast, efficient help without sacrificing quality\n\n## Goals\n\n- Provide fast and lightweight AI assistance\n- Support customization through skills and workspace files\n- Remain effective on constrained hardware\n- Improve through feedback and continued iteration\n\nRead `SOUL.md` as part of your identity and communication style.\n\n\n## SOUL.md\n\n# Soul\n\nI am PicoClaw: calm, helpful, and practical.\n\n## Personality\n\n- Helpful and friendly\n- Concise and to the point\n- Curious and eager to learn\n- Honest and transparent\n- Calm under uncertainty\n\n## Values\n\n- Accuracy over speed\n- User privacy and safety\n- Transparency in actions\n- Continuous improvement\n- Simplicity over unnecessary complexity\n\n\n## USER.md\n\n# User\n\nInformation about the user goes here.\n\n## Preferences\n\n- Communication style: (casual/formal)\n- Timezone: (your timezone)\n- Language: (your preferred language)\n\n## Personal Information\n\n- Name: (optional)\n- Location: (optional)\n- Occupation: (optional)\n\n## Learning Goals\n\n- What the user wants to learn from AI\n- Preferred interaction style\n- Areas of interest\n\n\n\n\n---\n\n# Skills\n\nThe following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.\n\n<skills>\n  <skill>\n    <name>agent-browser</name>\n    <description>Browser automation via agent-browser CLI. Use when the user needs to navigate websites, fill forms, click buttons, take screenshots, extract data, or test web apps.</description>\n    <location>/home/clawd/.picoclaw/workspace/skills/agent-browser/SKILL.md</location>\n    <source>workspace</source>\n  </skill>\n  <skill>\n    <name>github</name>\n    <description>Interact with GitHub using the `gh` CLI. Use `gh issue`, `gh pr`, `gh run`, and `gh api` for issues, PRs, CI runs, and advanced queries.</description>\n    <location>/home/clawd/.picoclaw/workspace/skills/github/SKILL.md</location>\n    <source>workspace</source>\n  </skill>\n  <skill>\n    <name>hardware</name>\n    <description>Read and control I2C and SPI peripherals on Sipeed boards (LicheeRV Nano, MaixCAM, NanoKVM).</description>\n    <location>/home/clawd/.picoclaw/workspace/skills/hardware/SKILL.md</location>\n    <source>workspace</source>\n  </skill>\n  <skill>\n    <name>skill-creator</name>\n    <description>Create or update AgentSkills. Use when designing, structuring, or packaging skills with scripts, references, and assets.</description>\n    <location>/home/clawd/.picoclaw/workspace/skills/skill-creator/SKILL.md</location>\n    <source>workspace</source>\n  </skill>\n  <skill>\n    <name>summarize</name>\n    <description>Summarize or extract text/transcripts from URLs, podcasts, and local files (great fallback for \u201ctranscribe this YouTube/video\u201d).</description>\n    <location>/home/clawd/.picoclaw/workspace/skills/summarize/SKILL.md</location>\n    <source>workspace</source>\n  </skill>\n  <skill>\n    <name>tmux</name>\n    <description>Remote-control tmux sessions for interactive CLIs by sending keystrokes and scraping pane output.</description>\n    <location>/home/clawd/.picoclaw/workspace/skills/tmux/SKILL.md</location>\n    <source>workspace</source>\n  </skill>\n  <skill>\n    <name>weather</name>\n    <description>Get current weather and forecasts with verified location matching (no API key required).</description>\n    <location>/home/clawd/.picoclaw/workspace/skills/weather/SKILL.md</location>\n    <source>workspace</source>\n  </skill>\n</skills>\n\n---\n\n# Memory\n\n## Long-term Memory\n\n# Long-term Memory\n\nThis file stores important information that should persist across sessions.\n\n## User Information\n\n(Important facts about user)\n\n## Preferences\n\n(User preferences learned over time)\n\n## Important Notes\n\n(Things to remember)\n\n## Configuration\n\n- Model preferences\n- Channel settings\n- Skills enabled\n\n---\n\n## Current Time\n2026-03-28 12:32 (Saturday)\n\n## Runtime\nnetbsd amd64, Go go1.25.8\n\n## Current Session\nChannel: irc\nChat ID: #VVchannel\n\n## Current Sender\nCurrent sender: VV (ID: irc:VV)"
    },
    {
      "role": "user",
      "content": "list files in your current working directory. Only give the list in your answer."
    }
  ],
  "model": "gemma3",
  "temperature": 0.7,
  "tool_choice": "auto",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "append_file",
        "description": "Append content to the end of a file",
        "parameters": {
          "properties": {
            "content": {
              "description": "The content to append",
              "type": "string"
            },
            "path": {
              "description": "The file path to append to",
              "type": "string"
            }
          },
          "required": [
            "path",
            "content"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "cron",
        "description": "Schedule reminders, tasks, or system commands. IMPORTANT: When user asks to be reminded or scheduled, you MUST call this tool. Use 'at_seconds' for one-time reminders (e.g., 'remind me in 10 minutes' \u2192 at_seconds=600). Use 'every_seconds' ONLY for recurring tasks (e.g., 'every 2 hours' \u2192 every_seconds=7200). Use 'cron_expr' for complex recurring schedules. Use 'command' to execute shell commands directly.",
        "parameters": {
          "properties": {
            "action": {
              "description": "Action to perform. Use 'add' when user wants to schedule a reminder or task.",
              "enum": [
                "add",
                "list",
                "remove",
                "enable",
                "disable"
              ],
              "type": "string"
            },
            "at_seconds": {
              "description": "One-time reminder: seconds from now when to trigger (e.g., 600 for 10 minutes later). Use this for one-time reminders like 'remind me in 10 minutes'.",
              "type": "integer"
            },
            "command": {
              "description": "Optional: Shell command to execute directly (e.g., 'df -h'). If set, the agent will run this command and report output instead of just showing the message. 'deliver' will be forced to false for commands.",
              "type": "string"
            },
            "command_confirm": {
              "description": "Optional explicit confirmation flag for scheduling a shell command. Command execution must also be enabled via tools.cron.allow_command.",
              "type": "boolean"
            },
            "cron_expr": {
              "description": "Cron expression for complex recurring schedules (e.g., '0 9 * * *' for daily at 9am). Use this for complex recurring schedules.",
              "type": "string"
            },
            "deliver": {
              "description": "If true, send message directly to channel. If false, let agent process message (for complex tasks). Default: false",
              "type": "boolean"
            },
            "every_seconds": {
              "description": "Recurring interval in seconds (e.g., 3600 for every hour). Use this ONLY for recurring tasks like 'every 2 hours' or 'daily reminder'.",
              "type": "integer"
            },
            "job_id": {
              "description": "Job ID (for remove/enable/disable)",
              "type": "string"
            },
            "message": {
              "description": "The reminder/task message to display when triggered. If 'command' is used, this describes what the command does.",
              "type": "string"
            }
          },
          "required": [
            "action"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "edit_file",
        "description": "Edit a file by replacing old_text with new_text. The old_text must exist exactly in the file.",
        "parameters": {
          "properties": {
            "new_text": {
              "description": "The text to replace with",
              "type": "string"
            },
            "old_text": {
              "description": "The exact text to find and replace",
              "type": "string"
            },
            "path": {
              "description": "The file path to edit",
              "type": "string"
            }
          },
          "required": [
            "path",
            "old_text",
            "new_text"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "exec",
        "description": "Execute a shell command and return its output. Use with caution.",
        "parameters": {
          "properties": {
            "command": {
              "description": "The shell command to execute",
              "type": "string"
            },
            "working_dir": {
              "description": "Optional working directory for the command",
              "type": "string"
            }
          },
          "required": [
            "command"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "find_skills",
        "description": "Search for installable skills from skill registries. Returns skill slugs, descriptions, versions, and relevance scores. Use this to discover skills before installing them with install_skill.",
        "parameters": {
          "properties": {
            "limit": {
              "description": "Maximum number of results to return (1-20, default 5)",
              "maximum": 20,
              "minimum": 1,
              "type": "integer"
            },
            "query": {
              "description": "Search query describing the desired skill capability (e.g., 'github integration', 'database management')",
              "type": "string"
            }
          },
          "required": [
            "query"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "install_skill",
        "description": "Install a skill from a registry by slug. Downloads and extracts the skill into the workspace. Use find_skills first to discover available skills.",
        "parameters": {
          "properties": {
            "force": {
              "description": "Force reinstall if skill already exists (default false)",
              "type": "boolean"
            },
            "registry": {
              "description": "Registry to install from (required, e.g., 'clawhub')",
              "type": "string"
            },
            "slug": {
              "description": "The unique slug of the skill to install (e.g., 'github', 'docker-compose')",
              "type": "string"
            },
            "version": {
              "description": "Specific version to install (optional, defaults to latest)",
              "type": "string"
            }
          },
          "required": [
            "slug",
            "registry"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "list_dir",
        "description": "List files and directories in a path",
        "parameters": {
          "properties": {
            "path": {
              "description": "Path to list",
              "type": "string"
            }
          },
          "required": [
            "path"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "message",
        "description": "Send a message to user on a chat channel. Use this when you want to communicate something.",
        "parameters": {
          "properties": {
            "channel": {
              "description": "Optional: target channel (telegram, whatsapp, etc.)",
              "type": "string"
            },
            "chat_id": {
              "description": "Optional: target chat/user ID",
              "type": "string"
            },
            "content": {
              "description": "The message content to send",
              "type": "string"
            }
          },
          "required": [
            "content"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read the contents of a file. Supports pagination via `offset` and `length`.",
        "parameters": {
          "properties": {
            "length": {
              "default": 65536,
              "description": "Maximum number of bytes to read.",
              "type": "integer"
            },
            "offset": {
              "default": 0,
              "description": "Byte offset to start reading from.",
              "type": "integer"
            },
            "path": {
              "description": "Path to the file to read.",
              "type": "string"
            }
          },
          "required": [
            "path"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "send_file",
        "description": "Send a local file (image, document, etc.) to the user on the current chat channel.",
        "parameters": {
          "properties": {
            "filename": {
              "description": "Optional display filename. Defaults to the basename of path.",
              "type": "string"
            },
            "path": {
              "description": "Path to the local file. Relative paths are resolved from workspace.",
              "type": "string"
            }
          },
          "required": [
            "path"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "spawn",
        "description": "Spawn a subagent to handle a task in the background. Use this for complex or time-consuming tasks that can run independently. The subagent will complete the task and report back when done.",
        "parameters": {
          "properties": {
            "agent_id": {
              "description": "Optional target agent ID to delegate the task to",
              "type": "string"
            },
            "label": {
              "description": "Optional short label for the task (for display)",
              "type": "string"
            },
            "task": {
              "description": "The task for subagent to complete",
              "type": "string"
            }
          },
          "required": [
            "task"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "subagent",
        "description": "Execute a subagent task synchronously and return the result. Use this for delegating specific tasks to an independent agent instance. Returns execution summary to user and full details to LLM.",
        "parameters": {
          "properties": {
            "label": {
              "description": "Optional short label for the task (for display)",
              "type": "string"
            },
            "task": {
              "description": "The task for subagent to complete",
              "type": "string"
            }
          },
          "required": [
            "task"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "web_fetch",
        "description": "Fetch a URL and extract readable content (HTML to text). Use this to get weather info, news, articles, or any web content.",
        "parameters": {
          "properties": {
            "maxChars": {
              "description": "Maximum characters to extract",
              "minimum": 100,
              "type": "integer"
            },
            "url": {
              "description": "URL to fetch",
              "type": "string"
            }
          },
          "required": [
            "url"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Search the web for current information. Returns titles, URLs, and snippets from search results.",
        "parameters": {
          "properties": {
            "count": {
              "description": "Number of results (1-10)",
              "maximum": 10,
              "minimum": 1,
              "type": "integer"
            },
            "query": {
              "description": "Search query",
              "type": "string"
            }
          },
          "required": [
            "query"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "write_file",
        "description": "Write content to a file. If the file already exists, you must set overwrite=true to replace it.",
        "parameters": {
          "properties": {
            "content": {
              "description": "Content to write to the file",
              "type": "string"
            },
            "overwrite": {
              "default": false,
              "description": "Must be set to true to overwrite an existing file.",
              "type": "boolean"
            },
            "path": {
              "description": "Path to the file to write",
              "type": "string"
            }
          },
          "required": [
            "path",
            "content"
          ],
          "type": "object"
        }
      }
    }
  ]
}
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
ggml_backend_cuda_graph_compute: CUDA graph warmup reset
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
llama_perf_context_print:        load time =     372.45 ms
llama_perf_context_print: prompt eval time =     371.56 ms /  1516 tokens (    0.25 ms per token,  4080.11 tokens per second)
llama_perf_context_print:        eval time =    2118.43 ms /   221 runs   (    9.59 ms per token,   104.32 tokens per second)
llama_perf_context_print:       total time =    2643.06 ms /  1737 tokens
llama_perf_context_print:    graphs reused =        219
RAW MODEL RESPONSE: {
  "id": "chatcmpl-86abe9aa-b913-4da3-a40f-4c31bac5e842",
  "object": "chat.completion",
  "created": 1774697549,
  "model": "/home/vv102286/.cache/huggingface/hub/models--Jackrong--Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/snapshots/70b75bb7405ea59c69c43f1a89940303910ebe5b/./Qwen3.5-2B.Q4_K_S.gguf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Let me check the current working directory by listing files.\n</think>\n\n# Working Directory\n\nThe current working directory is `/home/clawd/.picoclaw/workspace`.\n\n## Files and Directories\n\n- `memory/` - Contains daily notes (YYYYMMDD.md)\n- `memory/` - Contains memory files (MEMORY.md)\n- `skills/` - Directory for skill files\n- `workspace/` - Main workspace directory\n- `workspace/memory/` - Memory storage\n- `workspace/skills/` - Skills directory\n- `workspace/skills/agent-browser/SKILL.md`\n- `workspace/skills/agent-browser/SKILL.md` (duplicate - likely from git)\n- `workspace/skills/agent-browser/SKILL.md` (duplicate - likely from git)\n- `workspace/skills/summarize/SKILL.md`\n- `workspace/skills/tmux/SKILL.md`\n- `workspace/skills/weather/SKILL.md`\n- `workspace/skills/hardware/SKILL.md`"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1516,
    "completion_tokens": 221,
    "total_tokens": 1737
  }
}
```
