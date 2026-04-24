# APA Agent — AI-Driven Penetration Testing Assistant

**BSc (Hons) Cybercrime & IT Security — Final Year Project**  
**Student:** Elvis Omorusi (C00285993)  
**Supervisor:** Martin Tolan  

An intelligent agent that uses reinforcement learning (DQN/PPO) to autonomously
discover SQL Injection and Cross-Site Scripting vulnerabilities in web applications.
Trained and evaluated exclusively on intentionally vulnerable applications (DVWA)
in an isolated Docker environment.

---

## ⚠️ Ethical & Legal Notice

This tool is designed **exclusively for authorized security testing** in controlled
lab environments. It must **only** be used against applications you own or have
explicit written permission to test. Unauthorized use against any system is
**illegal** under the Computer Misuse Act and similar legislation.

All testing in this project uses:
- **DVWA** (Damn Vulnerable Web Application) — intentionally vulnerable
- **Docker isolation** — no network access to external systems
- **Authorized academic context** — supervised FYP with ethics approval

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Training Loop                         │
│                                                          │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  RL Agent   │───▶│  Gym Env     │───▶│   DVWA       │ │
│  │ (DQN/PPO)  │◀───│ (SQLi/XSS)   │◀───│  (Docker)    │ │
│  └────────────┘    └──────────────┘    └──────────────┘ │
│       │                    │                    │        │
│       │              ┌─────┴─────┐              │        │
│       │              │  Feature   │              │        │
│       │              │ Extractor  │              │        │
│       │              └─────┬─────┘              │        │
│       │              ┌─────┴─────┐              │        │
│       │              │ Response   │              │        │
│       │              │ Analyzer   │              │        │
│       │              └───────────┘              │        │
│       ▼                                         ▼        │
│  ┌────────────┐                        ┌──────────────┐ │
│  │TensorBoard │                        │  JSON Logs   │ │
│  └────────────┘                        └──────────────┘ │
│                                                          │
│  ┌────────────┐  (Optional Phase 3)                     │
│  │LLM Advisor │  Ollama + Mistral 7B                    │
│  └────────────┘                                          │
└──────────────────────────────────────────────────────────┘
```

## Project Structure

```
ai-pentest-assistant/
├── docker-compose.yml          # DVWA container setup
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration template
│
├── environments/               # Gymnasium environments
│   ├── base_env.py             # Abstract base (shared logic)
│   ├── sqli_env.py             # SQL Injection environment
│   ├── xss_env.py              # XSS environment
│   └── feature_extractors.py   # HTML → state vector conversion
│
├── agents/                     # Training & evaluation
│   ├── train.py                # Unified training script
│   └── evaluate.py             # Performance evaluation
│
├── payloads/                   # Payload definitions
│   ├── sqli_payloads.json      # SQLi payload families (8 categories)
│   └── xss_payloads.json       # XSS payload families (10 categories)
│
├── utils/                      # Core utilities
│   ├── dvwa_client.py          # DVWA HTTP client
│   ├── response_analyzer.py    # Vulnerability detection
│   └── logger.py               # Logging + TensorBoard callback
│
├── llm_advisor/                # LLM integration (optional)
│   ├── advisor.py              # Ollama/Mistral advisor
│   └── prompts/
│       └── pentest_system.md   # System prompt
│
├── configs/                    # Hyperparameter configs
│   ├── dqn_config.yaml
│   └── ppo_config.yaml
│
├── logs/                       # Training logs (generated)
├── models/                     # Saved models (generated)
└── results/                    # Evaluation results (generated)
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker & Docker Compose
- ~4GB disk space (for DVWA image + Python packages)

### 2. Setup

```bash
# Clone the project
git clone <your-repo-url>
cd ai-pentest-assistant

# Install Python dependencies
pip install -r requirements.txt

# Copy and edit configuration
cp .env.example .env

# Start DVWA in Docker
docker compose up -d

# Wait ~30 seconds for DVWA to initialize, then open
# http://localhost:8080 in a browser
# Login: admin / password
# Click "Create / Reset Database" on the setup page
```

### 3. Train an Agent

```bash
# Train DQN on SQL Injection (start here)
python -m agents.train --vuln sqli --algo dqn --timesteps 50000

# Train PPO on SQL Injection (for comparison)
python -m agents.train --vuln sqli --algo ppo --timesteps 50000

# Train on XSS
python -m agents.train --vuln xss --algo dqn --timesteps 50000

# Train both SQLi and XSS
python -m agents.train --vuln both --algo ppo --timesteps 50000

# Curriculum learning (low → medium → high difficulty)
python -m agents.train --vuln sqli --algo dqn --timesteps 90000 --curriculum

# Verbose mode (see every action)
python -m agents.train --vuln sqli --algo dqn --timesteps 10000 --verbose
```

### 4. Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir logs/tensorboard

# Open http://localhost:6006 in browser
# Key metrics: pentest/success_rate, pentest/mean_reward
```

### 5. Evaluate

```bash
# Evaluate a trained model
python -m agents.evaluate \
    --model models/<run_name>/sqli_dqn_final \
    --vuln sqli \
    --episodes 100 \
    --include-random

# Compare DQN vs PPO
python -m agents.evaluate \
    --model models/<dqn_run>/sqli_dqn_final models/<ppo_run>/sqli_ppo_final \
    --vuln sqli \
    --episodes 100 \
    --include-random \
    --output results/sqli_comparison.json
```

---

## Environment Design

### SQL Injection Environment

| Component | Details |
|-----------|---------|
| **Target** | DVWA SQLi page (`/vulnerabilities/sqli/`) |
| **Actions** | 10 discrete: baseline, 7 payload categories, encoded variants, report done |
| **State** | 18-dim vector: page features, response analysis, agent memory |
| **Success** | Data extracted from the `users` table |
| **Rewards** | +15 SQL error, +50 data leak, +100 multi-row extraction, -1/step |

### XSS Environment

| Component | Details |
|-----------|---------|
| **Target** | DVWA Reflected XSS page (`/vulnerabilities/xss_r/`) |
| **Actions** | 12 discrete: baseline, 10 payload categories, report done |
| **State** | 20-dim vector: page features, reflection analysis, agent memory |
| **Success** | Unescaped script/event handler reflected in response |
| **Rewards** | +10 reflection, +40 script execution, +80 confirmed XSS, -1/step |

---

## Experiments to Run (for your report)

### Experiment 1: DQN vs PPO on SQLi (Low Security)
Train both algorithms, compare success rate and learning speed.

### Experiment 2: DQN vs PPO on XSS (Low Security)
Same comparison for XSS.

### Experiment 3: Curriculum Learning
Train with progressive difficulty and show success rate at each level.

### Experiment 4: Random Baseline
Compare trained agents against a random agent to prove learning.

### Experiment 5: LLM Advisor (if time permits)
Compare RL-only vs RL+LLM on both vulnerability types.

---

## Key Metrics for Your Report

- **Learning curves** (TensorBoard): reward over episodes
- **Success rate**: % of episodes where vuln was found
- **Mean steps to success**: efficiency measure
- **Action distribution**: what strategies the agent learned
- **Cross-difficulty transfer**: does Low training help on Medium/High?

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| DVWA not accessible | `docker compose up -d`, wait 30s, check http://localhost:8080 |
| "Login failed" | Visit http://localhost:8080/setup.php, click "Create / Reset Database" |
| Agent never finds vulns | Check DVWA security is "low", increase `exploration_fraction` |
| Training too slow | Reduce `max_steps`, ensure DVWA Docker has enough resources |
| TensorBoard empty | Check `--logdir` path matches where logs are saved |
| Import errors | Run from project root: `python -m agents.train` (not `python agents/train.py`) |

---

## License & Ethics

This project is for **educational purposes only** as part of an authorized
academic final year project. It implements security testing techniques against
deliberately vulnerable applications in isolated environments.

The tools, techniques, and code in this repository must not be used against
any system without explicit written authorization from the system owner.
