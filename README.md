# Gerador de grafos HoVer com Cognee

Este projeto le o CSV em `input/`, processa cada linha do dataset HoVer e grava
um JSON por item em `output/`.

O CSV HoVer traz duas colunas parecidas:

- `evidence_text`: texto das evidencias, em JSON, usado para gerar
  `grafo_evidencia`.
- `evidence`: metadados do exemplo, como `supporting_facts` e `num_hops`.

O gerador transforma `evidence_text` em texto plano antes de chamar o Cognee,
mantem `evidencia` por compatibilidade com o exportador e tambem grava
`evidence_text`, `num_hops` e `hover_metadata` no JSON final.

Cada JSON contem:

- `dataset`
- `id`
- `split`
- `claim`
- `evidencia`
- `evidence_text`
- `label`
- `num_hops`, quando presente
- `hover_metadata`, quando presente
- `grafo_claim.nodes`
- `grafo_claim.edges`
- `grafo_evidencia.nodes`
- `grafo_evidencia.edges`

Os grafos sao gravados em formato enxuto para uso posterior em GNN:

```json
{
  "nodes": [
    {
      "id": 0,
      "original_id": "id-original-do-cognee",
      "text": "texto ou nome do no",
      "type": "tipo do no"
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "type": "tipo-da-relacao"
    }
  ]
}
```

O Cognee usado em runtime e a copia Petrobras adaptada em `vendor/`. O `.env`,
os certificados `.pem` e o cache do `tiktoken` foram importados do projeto
funcional `cognee-grafo`.

## Uso

Crie o ambiente virtual e instale as dependencias pelo JFrog Petrobras:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://jfrog.petrobras.dev.br/artifactory/api/pypi/pypi-group-all/simple --trusted-host jfrog.petrobras.dev.br
```

Se o `pip` reclamar de caminho invalido para certificado TLS, aponte para o
bundle importado neste projeto e rode o `pip install` novamente:

```powershell
$env:REQUESTS_CA_BUNDLE=(Resolve-Path petrobras-cert-bundle.pem).Path
$env:SSL_CERT_FILE=$env:REQUESTS_CA_BUNDLE
```

Valide caminhos e primeiros arquivos planejados sem chamar o Cognee:

```powershell
.\.venv\Scripts\python.exe run.py --dry-run --limit 2
```

Processe as primeiras 10 linhas:

```powershell
.\.venv\Scripts\python.exe run.py --limit 10
```

Processe tudo:

```powershell
.\.venv\Scripts\python.exe run.py
```

Exporte os JSONs aprovados na qualidade para uma pasta de execucao:

```powershell
.\.venv\Scripts\python.exe src\export.py
```

O modo vem de `EXPORT_PROCESS` na `.env`. Valores aceitos: `export total`,
`export com textDocument` e `export limpo`. Cada execucao cria
`exports/<data-hora>_<processo>/json` com copia dos JSONs aprovados e
`exports/<data-hora>_<processo>/parquet` com `graphs.parquet`,
`nodes.parquet` e `edges.parquet`. O exportador usa a coluna `split` do CSV
HoVer para preencher a coluna `split` dos Parquets e inclui `num_hops` em
`graphs.parquet` quando o metadado existir.

A coluna `graph_id` liga as tres tabelas; ela e derivada do nome do JSON, entao
continua unica mesmo quando o mesmo `id` aparece em mais de uma linha.

Para retomar uma execucao, rode o mesmo comando novamente. Arquivos JSON ja
existentes sao pulados, exceto quando `--overwrite` for usado.

Opcoes uteis:

```powershell
.\.venv\Scripts\python.exe run.py --start-row 100 --limit 50
.\.venv\Scripts\python.exe run.py --overwrite --limit 5
.\.venv\Scripts\python.exe run.py --continue-on-error
.\.venv\Scripts\python.exe run.py --max-attempts 3 --continue-on-error
```

Por padrao cada linha e tentada ate 3 vezes antes de ser marcada como falha.
Entre tentativas, o estado temporario do Cognee e limpo para reduzir efeitos de
uma execucao parcial.

Erros `content_filter` do Azure OpenAI nao sao repetidos pelo script, porque
tendem a falhar novamente com o mesmo prompt. Nesses casos, o item recebe um
JSON com `erro.category = "content_filter"` e `erro.motivo` descrevendo a
categoria bloqueada quando essa informacao estiver disponivel.

Por padrao o estado temporario do Cognee e limpo entre o grafo do claim e o da
evidencia, para evitar contaminacao entre grafos. Use
`--no-clean-between-graphs` somente se quiser priorizar velocidade e aceitar
estado compartilhado por dataset.
