# Text2SQL Finetuner
This project is a case study to see the feasibility of training LLM up to 3B parameter in a limited resource. 

## Prequisites
- [Docker Installation](https://docs.docker.com/engine/install/)
- [Dev Container Installation](https://code.visualstudio.com/docs/devcontainers/containers#_getting-started)
- GPU with RAM at least 6GB

## Quickstart
- Open IDE that supports Dev Container, e.g. Visual Studio Code.
- Press `Ctrl` + `Shift` + `P` and pick `Dev Containers: Rebuild and Reopen in Container`
- Open `.ipynb` file and start finetune your model!

## Available Models
We experiment with:
- `tiiuae/Falcon3-1B-Base` - https://huggingface.co/tiiuae/Falcon3-1B-Base
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` (3 epochs) - https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` (12 epochs) - https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct

## Report
See `report.md` or `Text2SQL_Report_from_Malvin.pdf` for insights in this case study. 
