# WONCA RAG

Pipeline multimodelo para análise automatizada de artigos de revisão integrativa, com RAG temporal, classificação S/N/D, rastreabilidade de evidências e comparação posterior com revisores humanos.

## Princípios

- Processamento em ordem cronológica.
- Nenhum artigo poderá recuperar informação de artigos posteriores.
- Avaliações humanas ficam separadas da inferência.
- O RAG será compartilhado entre os modelos sempre que possível.
- Cada modelo terá seu próprio histórico conceitual.
- Código principal em `src/`, notebooks para execução e análise.

## Setup no VS Code

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_setup.py
pytest
```

Copie `.env.example` para `.env` apenas quando começarmos a integrar APIs.
Nunca publique `.env`, chaves, PDFs privados ou avaliações humanas no Git.
