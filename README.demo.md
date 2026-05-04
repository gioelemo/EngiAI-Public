# EngiAI — Minimal Demo (Windows / WSL2)

A stripped-down build of the EngiAI chatbot for live demos. Runs **two containers** (Streamlit UI + Postgres), no RAG, no 3D-printer integration, and no bundled HPC services. The chatbot can still submit jobs to an external HPC cluster (e.g. Euler) if SSH credentials are provided at runtime via the Settings page — see the optional Euler / WandB section under step 4.

## Prerequisites

- Windows 10 (build 19041+) or Windows 11
- [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) enabled
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the WSL2 backend enabled (Settings → General → *Use the WSL 2 based engine*)
- Git (inside WSL: `sudo apt install git`)

## 1. Install WSL (one time)

From PowerShell **as Administrator**:

```powershell
wsl --install
```

Reboot, then open the new Ubuntu terminal and finish the first-run setup.

## 2. Clone the repo **inside** the WSL filesystem

Do **not** clone into `/mnt/c/...` — Docker builds are dramatically slower across the Windows/WSL mount boundary and line endings can cause build issues.

```bash
cd ~
git clone git@github.com:gioelemo/EngiAI.git
cd EngiAI
```

## 3. Configure environment

```bash
cp .env.demo.example .env
nano .env    # fill in GOOGLE_API_KEY or OPENAI_API_KEY; TAVILY_API_KEY is optional
```

At minimum set the API key for the LLM provider you want to use, and leave `POSTGRES_PASSWORD` as-is or pick your own.

- **Google (default)** — set `GOOGLE_API_KEY`. `LLM_MODEL` is already pointed at `google_genai:...` in the example file, no further change needed.
- **OpenAI** — set `OPENAI_API_KEY`, then also switch `LLM_MODEL` to an OpenAI model: comment out the `google_genai:` line and uncomment the `openai:` line in `.env`.

**Optional** — to train models on the Euler cluster and track runs on WandB, also set the following in `.env` (create a WandB entity first if you don't have one):

- `WANDB_API_KEY`
- `HF_TOKEN`
- `USE_WANDB=True` — required (exact casing) to download trained models back from WandB; without it the chatbot will reply *"WandB is not enabled. Set USE_WANDB=True in your environment variables."*

## 4. Launch

```bash
docker compose -f docker-compose.demo.yml up -d --build
```

First build takes a few minutes (10-15 min, Python deps). Subsequent starts are instant.

Then open <http://localhost:8501> in your Windows browser.

### Optional — Euler / WandB setup

If you want to train models on the Euler cluster and track runs on WandB, configure the following in the **Settings** page of the UI:

- **HPC Connection** — select *Password Authentication* and fill in the hostname, username, and password. Click **Test Connection**; once it succeeds, click **Save Credentials**.
- **Email for Job Notifications**, **WandB Entity**, and **WandB Project** — set these too. You can leave the other paths at their defaults if you're using the `mavt-ide-euler` user.

You can now ask the chatbot to generate a SLURM script, submit it to the cluster, and get back an example design produced by the freshly trained model.


## 5. Stop / reset

```bash
# Stop containers (keeps conversation history)
docker compose -f docker-compose.demo.yml down

# Stop AND wipe the demo database
docker compose -f docker-compose.demo.yml down -v
```

## What's different from the full stack?

| Feature | Full stack | Demo |
| --- | --- | --- |
| Streamlit chatbot | yes | **yes** |
| Postgres | yes | **yes** (separate volume) |
| MMORE RAG service | yes | no (`SKIP_MMORE=true`) |
| Prusa MCP server | yes | no (`SKIP_MCP=true`) |
| ArXiv agent | yes | no (`SKIP_ARXIV=true`) |
| SSH / HPC mounts | yes | no |
| Papers directory | yes | no |
| Image tags | `engiai-chatbot` | `engiai-demo-chatbot` |
| Volume | `engiai_postgres_data` | `engiai_demo_data` |

The demo compose project is named `engiai-demo`, so it can run **side-by-side** with the full stack (the only collision is host port `8501` — stop one before starting the other, or edit the port mapping).

## Troubleshooting

- **`docker: command not found`** — Docker Desktop not running, or WSL integration not enabled for your distro (Docker Desktop → Settings → Resources → WSL Integration).
- **Build is extremely slow** — you likely cloned under `/mnt/c/...`. Move the repo into the WSL home (`~`).
- **`port is already allocated`** — another service is using 8501. Either stop it, or change `"8501:8501"` in `docker-compose.demo.yml` to e.g. `"8502:8501"`.
- **Chatbot container restarts in a loop** — check logs with `docker logs engiai-demo-chatbot`. Most common cause: missing or malformed `GOOGLE_API_KEY` in `.env`.
- **Line-ending errors on shell scripts** — ensure `git config --global core.autocrlf input` inside WSL before cloning.
