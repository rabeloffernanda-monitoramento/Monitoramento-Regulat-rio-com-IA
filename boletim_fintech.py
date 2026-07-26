"""
Boletim diário de notícias regulatórias para fintech (CloudWalk).

Consome o servidor MCP Brasil (https://github.com/Mcp-Brasil/mcp-brasil) via
protocolo MCP e usa a API da Anthropic (Claude) para decidir quais tools
chamar, filtrar o que é relevante, classificar cada item nas categorias
abaixo e atribuir um nível de relevância (Alta/Média/Baixa) com a
justificativa do impacto para a CloudWalk. O resultado final é exportado
em CSV, ordenado por relevância.

Requisitos:
    pip install mcp-brasil mcp anthropic

Variáveis de ambiente:
    ANTHROPIC_API_KEY   -> obrigatória
    (as chaves do mcp-brasil - TRANSPARENCIA_API_KEY etc - não são
     necessárias para as tools usadas aqui)
"""

import asyncio
import csv
import json
import os
import shutil
from datetime import date
from pathlib import Path

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CATEGORIAS = [
    "Banco Central, CMN e CVM",
    "Meios de pagamento, Pix e Open Finance",
    "Cartões de crédito e concessão de crédito",
    "Instituições financeiras e instituições de pagamento",
    "Prevenção a fraudes, segurança cibernética e inteligência artificial",
    "Defesa do consumidor e alterações legislativas",
]

# O servidor MCP Brasil não expõe as ~300 tools de fontes de dados
# diretamente: ele expõe só 7 "meta-tools" de descoberta/execução, e as
# tools de dados de verdade (noticias, diario_oficial, bacen, brasilapi
# etc.) são encontradas em tempo de execução via `search_tools` /
# `recomendar_tools` e chamadas via `call_tool`. Por isso não filtramos
# por nome aqui: repassamos as meta-tools inteiras para o Claude decidir.
TOOLS_MCP_BRASIL = (
    "listar_features",
    "recomendar_tools",
    "planejar_consulta",
    "listar_datasets_disponiveis",
    "executar_lote",
    "search_tools",
    "call_tool",
)

# Fontes de dados que nos interessam (usadas só para orientar o Claude no
# prompt; os nomes exatos das tools internas são descobertos via
# search_tools/recomendar_tools, não fixados aqui).
FONTES_DE_INTERESSE = (
    "notícias agregadas (Câmara, Senado, Agência Brasil, BCB)",
    "Diário Oficial da União e Querido Diário (normativos de CVM/CMN)",
    "Banco Central (séries, normativos, BCB Olinda)",
    "BrasilAPI (Pix, bancos, dados de apoio)",
)

# Critérios de classificação de relevância para a CloudWalk (IP / SCFI).
CRITERIOS_RELEVANCIA = """
- Alta: mudanças nas normas de capital, novas regras obrigatórias de
  segurança cibernética ou alterações nas tarifas/regras do Pix e Open
  Finance.
  Por quê: instituições S4 possuem estruturas menores e qualquer mudança
  normativa exige rápida adaptação para manter a conformidade.
- Média: novas ferramentas de IA para detecção de fraude ou tendências de
  consumo no setor de crédito.
  Por quê: impactam a competitividade e a eficiência operacional, mas não
  geram risco imediato de sanção regulatória.
- Baixa: notícias macroeconômicas genéricas ou variações de índices que não
  alteram as projeções de curto prazo.
  Por quê: embora importantes para o contexto, não exigem uma ação imediata
  do board da IP ou SCFI.
"""

RELEVANCIAS_VALIDAS = ("Alta", "Média", "Baixa")

SYSTEM_PROMPT = f"""Você é um analista de compliance regulatório da CloudWalk
(instituição de pagamento / SCFI), responsável por montar o boletim diário
regulatório do time.

O servidor MCP Brasil não expõe as fontes de dados diretamente como tools
individuais. Para usá-lo, siga este fluxo:
1. Use `recomendar_tools` (ou `search_tools`) com uma pergunta em linguagem
   natural para descobrir quais tools internas existem para o que você
   precisa (ex.: "notícias recentes do Banco Central sobre Pix",
   "publicações do Diário Oficial sobre CVM e CMN").
2. É OBRIGATÓRIO fazer pelo menos uma busca dedicada para CADA uma destas
   fontes, nesta ordem, antes de escrever a resposta final -- não pule
   nenhuma, mesmo que já tenha achado bastante conteúdo em outra fonte:
{chr(10).join(f"   {i}. {f}" for i, f in enumerate(FONTES_DE_INTERESSE, start=1))}
3. Depois de identificar a tool certa (com `recomendar_tools`/`search_tools`),
   chame-a de fato usando `call_tool`, passando o nome exato da tool
   encontrada e os argumentos que ela espera.
4. Se precisar rodar várias tools de uma vez, pode usar `executar_lote`.
5. Se, mesmo depois de buscar dedicadamente em uma fonte, ela não retornar
   nada relevante às categorias abaixo, tudo bem -- só não pule a etapa de
   busca em si. O Banco Central (fonte 3) é especialmente importante:
   normativos do BCB saem quase todo dia e não podem ser ignorados.

Busque notícias e publicações oficiais do dia (últimas 24-48h) relacionadas
a estas categorias:

{chr(10).join(f"- {c}" for c in CATEGORIAS)}

Regras de seleção:
- Priorize fontes oficiais (BCB, DOU/CVM/CMN, Câmara, Senado) e Agência Brasil.
- Ignore qualquer item fora dessas categorias.
- Não invente itens: se uma tool não retornar nada relevante para uma
  categoria, apenas não preencha essa categoria.
- Seja eficiente: não é necessário detalhar cada proposição legislativa
  individualmente com múltiplas chamadas de tool. Priorize notícias e
  publicações já resumidas nas fontes agregadas antes de investigar
  proposições uma a uma. Use no máximo 3 chamadas de detalhamento de
  proposições individuais (ex.: `camara_detalhar_proposicao`) no total --
  depois disso, siga para as próximas fontes obrigatórias da lista acima.

Para cada item relevante, extraia:
- categoria: uma das categorias acima, use o texto exato.
- titulo, fonte, data (AAAA-MM-DD), link.
- resumo: objetivo, até 2 frases, explicando o impacto prático para uma
  fintech.
- relevancia: classifique como "Alta", "Média" ou "Baixa" segundo os
  critérios abaixo, específicos para a CloudWalk (IP / SCFI, instituição
  S4):
{CRITERIOS_RELEVANCIA}
- justificativa: uma frase objetiva explicando por que a notícia é
  relevante (ou não) especificamente para a CloudWalk, usando a lógica dos
  critérios acima (ex.: risco de conformidade, competitividade/eficiência,
  ou apenas contexto macro).

Ao final, responda APENAS com um JSON válido, sem texto antes ou depois, e
sem envolver em blocos de código markdown (nada de ```json ou ```). A
resposta deve começar direto com "{" e terminar com "}", no formato:
{{"itens": [{{"categoria": "...", "titulo": "...", "fonte": "...", "data": "AAAA-MM-DD", "link": "...", "resumo": "...", "relevancia": "Alta|Média|Baixa", "justificativa": "..."}}]}}
"""

MODEL = "claude-sonnet-5"


def _localizar_uvx() -> str:
    encontrado = shutil.which("uvx")
    if encontrado:
        return encontrado
    caminho_padrao = os.path.expanduser("~/.local/bin/uvx")
    if os.path.exists(caminho_padrao):
        return caminho_padrao
    raise RuntimeError("uvx nao encontrado no PATH nem em ~/.local/bin/uvx")


async def coletar_boletim() -> list[dict]:
    server_params = StdioServerParameters(
        command=_localizar_uvx(),
        args=["--from", "mcp-brasil", "python", "-m", "mcp_brasil.server"],
        env=dict(os.environ),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema,
                }
                for t in tools_result.tools
                if t.name in TOOLS_MCP_BRASIL
            ]

            if not tools:
                nomes_encontrados = [t.name for t in tools_result.tools]
                raise RuntimeError(
                    "Nenhuma tool esperada foi encontrada no servidor MCP "
                    f"Brasil. Tools disponíveis: {nomes_encontrados}. "
                    "O servidor pode ter mudado a interface exposta -- "
                    "rode debug_tools.py para conferir os nomes atuais."
                )

            client = Anthropic()  # usa ANTHROPIC_API_KEY do ambiente
            messages = [
                {
                    "role": "user",
                    "content": f"Monte o boletim de hoje ({date.today().isoformat()}).",
                }
            ]

            # loop de agente: Claude pode chamar tools várias vezes antes de
            # entregar a resposta final em JSON. Com o fluxo de duas etapas
            # (buscar tool -> chamar tool) o número de turnos tende a ser
            # maior, então limitamos por segurança em vez de deixar infinito.
            response = None
            MAX_TURNOS = 40
            for turno in range(1, MAX_TURNOS + 1):
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=8000,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=messages,
                )
                messages.append({"role": "assistant", "content": response.content})

                chamadas = [b.name for b in response.content if getattr(b, "type", None) == "tool_use"]
                if chamadas:
                    print(f"  [turno {turno}] chamando: {', '.join(chamadas)}")
                else:
                    print(f"  [turno {turno}] resposta final (stop_reason={response.stop_reason})")

                if response.stop_reason != "tool_use":
                    break

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = await session.call_tool(block.name, block.input or {})
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": [
                                    {"type": "text", "text": c.text}
                                    for c in result.content
                                    if hasattr(c, "text")
                                ],
                            }
                        )
                messages.append({"role": "user", "content": tool_results})
            else:
                raise RuntimeError(
                    f"Boletim não finalizou após {MAX_TURNOS} turnos de "
                    "tool use -- verifique se o modelo está em loop."
                )

    texto_final = "".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    )

    if not texto_final.strip():
        raise RuntimeError(
            "O Claude terminou sem produzir texto de resposta final "
            f"(stop_reason={response.stop_reason!r}). Conteúdo bruto da "
            f"última resposta: {response.content!r}\n"
            "Causas comuns: estourou max_tokens antes de fechar o JSON, ou "
            "o modelo parou por outro motivo. Tente rodar de novo -- se "
            "persistir, pode ser necessário reduzir o escopo de tools "
            "usadas por turno."
        )

    # às vezes o modelo envolve o JSON num bloco de código markdown
    # (```json ... ```) mesmo quando instruído a não fazer isso -- removemos
    # esse envelope antes de tentar interpretar.
    texto_json = texto_final.strip()
    if texto_json.startswith("```"):
        texto_json = texto_json.split("\n", 1)[1] if "\n" in texto_json else texto_json
        if texto_json.rstrip().endswith("```"):
            texto_json = texto_json.rstrip()[: -3]
        texto_json = texto_json.strip()

    try:
        dados = json.loads(texto_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Resposta final do Claude não é um JSON válido: {e}\n"
            f"Texto recebido (primeiros 1000 caracteres):\n{texto_final[:1000]}"
        ) from e

    itens = dados.get("itens", [])

    # normaliza/valida a relevância retornada pelo modelo; itens fora do
    # padrão esperado caem em "Baixa" para não travar o boletim.
    for item in itens:
        relevancia = str(item.get("relevancia", "")).strip().capitalize()
        item["relevancia"] = relevancia if relevancia in RELEVANCIAS_VALIDAS else "Baixa"

    ordem = {"Alta": 0, "Média": 1, "Baixa": 2}
    itens.sort(key=lambda item: ordem.get(item["relevancia"], 3))
    return itens


def salvar_csv(itens: list[dict], caminho: Path) -> None:
    # ordem e nomes de coluna conforme solicitado pelo time de compliance
    campos = [
        ("titulo", "Título"),
        ("data", "Data"),
        ("fonte", "Fonte"),
        ("link", "Link"),
        ("relevancia", "Classificação de Relevância"),
        ("resumo", "Resumo"),
        ("justificativa", "Possível Impacto para a CloudWalk"),
    ]
    chaves = [c[0] for c in campos]
    cabecalho = [c[1] for c in campos]

    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cabecalho)
        for item in itens:
            writer.writerow([item.get(chave, "") for chave in chaves])


def main() -> None:
    itens = asyncio.run(coletar_boletim())
    saida = Path(f"boletim_{date.today().isoformat()}.csv")
    salvar_csv(itens, saida)
    contagem = {"Alta": 0, "Média": 0, "Baixa": 0}
    for item in itens:
        contagem[item["relevancia"]] = contagem.get(item["relevancia"], 0) + 1
    print(
        f"Boletim salvo em {saida} ({len(itens)} itens: "
        f"{contagem['Alta']} alta, {contagem['Média']} média, {contagem['Baixa']} baixa)"
    )


if __name__ == "__main__":
    main()
