# CFQD — Rodada completa: resultados e preocupações

*Branch `marcelo-21nov` · 2026-11-21, atualizado em 2026-11-22 após a re-rodada · Versão em inglês: [full_run_results.en.md](full_run_results.en.md)*

As seções 2.1–2.6 usam `results/full/summary/*.csv` (700 configurações, 70.000 linhas instância × orçamento) e
`results/full/manifest.jsonl`, além das saídas salvas no notebook 03.
**Atualização de 2026-11-22:** as 700 configurações foram re-rodadas com o código commitado (`results/full_rerun`),
já com a correção do DiCE-random (C9), e a simulação completa foi rodada (`results/simulation`). A re-rodada reproduz
os resultados originais (C4). A seção 2.7, a seção 2.8 e as linhas do DiCE-random na §2.3 vêm dessas rodadas novas.
Salvo indicação em contrário, os números são do **maior orçamento (10.000 consultas)**. Cada
(dataset, modelo, método, seed) é primeiro agregado pela média das suas 20 instâncias; depois, pela média entre
seeds, modelos e datasets.

## Resumo

1. **O MAP-Elites tem a maior cobertura (coverage) em todos os 150 blocos (dataset, modelo, seed).** Ele preenche, em
   média, 41% das células viáveis do grid estratégia × esforço. O melhor baseline preenche 6%. A diferença é de 11× a
   30× em German Credit, Adult, HELOC e GMC, e de só 2,2× no COMPAS, cujo grid tem 10 células.
2. **Nenhum método violou restrições** (0 violações em todas as configurações). O MAP-Elites acha um contrafactual
   válido para 97,7% das instâncias. O Wachter (só LR e NN) acha para 98,5%. O FACE é o mais fraco, com 85,5%.
3. **O contrafactual mais próximo do MAP-Elites costuma ser mais próximo que o dos baselines.** Ele vence
   DiCE-genetic, Wachter e FACE em 99% das instâncias pareadas, e DiCE-random em 84%.
4. **Essas vitórias ainda não são uma resposta justa à RQ1.** A cobertura é limitada pelo número de contrafactuais
   que o método devolve: 1 para Wachter e FACE, no máximo 10 para DiCE, 85 em média para o MAP-Elites. O MAP-Elites
   também é o único método que enxerga o grid e o único cujo fitness é a distância usada na avaliação. Além disso,
   ele gasta o orçamento inteiro, enquanto os baselines param após uma mediana de 1 a 2.031 consultas. Por
   contrafactual devolvido, o DiCE é tão diverso em estratégias quanto o MAP-Elites, ou mais (preocupação C1).
5. **A mistura de versões de código não muda os resultados** (C4, resolvida em 2026-11-22). 438 das 700
   configurações tinham rodado com código-fonte mais antigo. Re-rodando as 700 com o código commitado, 696 dão
   resultados idênticos. As outras 4 são do FACE, e cada uma difere em uma única instância.
6. **A simulação completa confirma o resultado das rotas.** O MAP-Elites acha cerca do dobro das rotas conhecidas que o
   DiCE-genetic acha (0,66 vs. 0,34 com 10.000 consultas), em todos os blocos (modelo, seed). É o único método que
   chega com frequência às rotas distantes. Duas ressalvas: embaralhar o grid custa 0,07 de recall, e 42% dos seus
   contrafactuais invertem o modelo sem cumprir nenhuma rota verdadeira (C18).
7. **A RQ2 (QD vs. QD) ainda não pode ser respondida.** NS, NSLC, surprise e MOC não estão implementados.
   Plausibilidade também não é medida, então a parte "sem perder plausibilidade" da RQ1 continua em aberto.

---

## 1. O que foi rodado

| | |
| --- | --- |
| Datasets | German Credit, Adult, HELOC, Give Me Some Credit (GMC), COMPAS (acionabilidade de Guidotti). O COMPAS estrito é pulado: não tem atributo acionável. |
| Modelos | LR, RF, NN (PyTorch), um modelo por seed |
| Métodos | MAP-Elites (único método QD até agora), DiCE-genetic, DiCE-random, Wachter (só LR e NN), FACE-kNN |
| Seeds | 10 (0–9). Cada seed controla o split, o treino e a busca. |
| Instâncias | 20 por configuração, sorteadas entre as linhas de teste **preditas como desfavoráveis por aquele modelo**. Todos os métodos usam as mesmas 20 instâncias dentro de um (dataset, modelo, seed), então as comparações entre métodos são pareadas. |
| Orçamentos | 500, 1.000, 2.000, 5.000 e 10.000 consultas ao modelo. Cada orçamento é uma execução independente com a mesma seed de busca. |
| Configurações | 5 datasets × 3 modelos × 5 métodos × 10 seeds − 50 (Wachter em RF) = **700** |
| Custo | 36,5 horas de CPU registradas (675 configurações com tempo), concluídas entre 2026-11-20 17:49 e 2026-11-21 00:40 UTC |

### Tamanho do grid estratégia × esforço

As faixas de esforço usam os limites `[0.1, 0.25, 0.5, 1.0]`, o que dá 5 faixas. A cobertura é medida sobre as
células **viáveis**.

| Dataset | Atributos | Grupos (g) | Estratégias viáveis | Células viáveis |
| --- | ---: | ---: | ---: | ---: |
| German Credit | 20 | 4 | 80 | 400 |
| Adult | 12 | 4 | 35 | 175 (de 400) |
| HELOC | 26 | 4 | 80 | 400 |
| GMC | 10 | 3 | 26 | 130 |
| COMPAS (Guidotti) | 9 | 1 | 2 | 10 |

### Modelos explicados (acurácia de teste, seed 0, de `reports/accuracies.csv`)

| Dataset | LR | RF | NN | Classe majoritária | Acurácia balanceada (LR / RF / NN) |
| --- | ---: | ---: | ---: | ---: | --- |
| German Credit | 0,733 | 0,717 | 0,713 | 0,700 | 0,68 / 0,63 / 0,60 |
| Adult | 0,849 | 0,865 | 0,857 | 0,752 | 0,77 / 0,78 / 0,78 |
| HELOC | 0,717 | 0,724 | 0,722 | 0,520 | 0,72 / 0,72 / 0,72 |
| GMC | 0,935 | 0,935 | 0,934 | 0,931 | 0,57 / 0,58 / 0,58 |
| COMPAS (Guidotti) | 0,672 | 0,675 | 0,673 | 0,545 | 0,66 / 0,67 / 0,66 |

---

## 2. Resultados

### 2.1 Visão geral com orçamento 10.000

Todos os datasets e modelos. O Wachter cobre só LR e NN.

| Método | Sucesso | Cobertura | Células preenchidas | Estratégias | CFs devolvidos | Mediana da distância do mais próximo | Mediana de L0 | Mudanças reversíveis¹ | Mediana de consultas usadas | Segundos / instância |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **MAP-Elites** | 0,977 | **0,409** | **85,5** | **27,1** | 85,5 | **0,106** | 3,5 | 64% | 10.000 | 5,2 |
| DiCE-genetic | 0,942 | 0,056 | 4,3 | 3,5 | 8,9 | 0,410 | 4,4 | 68% | ~830 | 10,4 |
| DiCE-random | 0,943 | 0,060 | 5,1 | 3,9 | 9,1 | 0,262 | **1,7** | **33%** | 2.031 | 0,6 |
| Wachter | **0,985** | 0,022 | 1,0 | 1,0 | 1,0 | 0,535 | 5,0 | 53% | 9 | 0,6 |
| FACE | 0,855 | 0,021 | 0,9 | 0,9 | 0,9 | 0,667 | 5,0 | 69% | 1 | 0,1 |

¹ Redundância do CARLA dividida por L0: a fração dos atributos alterados que poderiam ser revertidos sem perder a
classe desejada.

**Comentário.** O MAP-Elites domina todas as métricas de diversidade (cobertura, células, estratégias) por uma ordem
de grandeza. Seu contrafactual mais próximo também é o mais próximo de todos os métodos. Isso não sai às custas de
validade: o sucesso fica no mesmo nível dos baselines, e todo contrafactual devolvido respeita as restrições. O preço
é o custo: o MAP-Elites sempre gasta o orçamento inteiro. O DiCE-random, que em tese é o "piso", é o método mais
esparso (1,7 atributos alterados) e o que tem menos mudanças desnecessárias. Ele também supera o DiCE-genetic em
cobertura e proximidade (ver C9).

### 2.2 Cobertura por dataset

Média sobre modelos e seeds, com orçamento 10.000.

| Dataset | MAP-Elites | DiCE-genetic | DiCE-random | Wachter | FACE | MAP-Elites ÷ melhor baseline | Cobertura máxima com 10 CFs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| German Credit | **0,340** | 0,019 | 0,018 | 0,002 | 0,002 | 18× | 0,025 |
| Adult | **0,371** | 0,019 | 0,035 | 0,006 | 0,005 | 11× | 0,057 |
| HELOC | **0,392** | 0,011 | 0,013 | 0,002 | 0,002 | 30× | 0,025 |
| GMC | **0,505** | 0,035 | 0,039 | 0,008 | 0,007 | 13× | 0,077 |
| COMPAS | **0,438** | 0,199 | 0,194 | 0,094 | 0,090 | 2,2× | 1,000 |

**Comentário.** A última coluna é a cobertura máxima que um método que devolve 10 contrafactuais conseguiria atingir.
Em German Credit e HELOC, o DiCE nunca passaria de 2,5% de cobertura, nem se cada um dos seus 10 contrafactuais
caísse numa célula diferente. O tamanho da diferença de cobertura é, portanto, definido principalmente pelo tamanho
do conjunto devolvido, e não pela qualidade da busca (C1). O COMPAS tem o problema oposto: com 1 grupo, o grid tem
só 2 estratégias e não consegue expressar diversidade de estratégias (C11). A variação entre seeds é pequena: o
desvio-padrão da cobertura do MAP-Elites dentro de um (dataset, modelo) fica entre 0,02 e 0,11.

### 2.3 Curvas de orçamento

Média sobre todos os datasets, modelos e seeds.

| Orçamento | 500 | 1.000 | 2.000 | 5.000 | 10.000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAP-Elites: cobertura | 0,240 | 0,292 | 0,335 | 0,381 | 0,409 |
| MAP-Elites: células preenchidas | 39,4 | 52,6 | 64,2 | 77,1 | 85,5 |
| MAP-Elites: sucesso | 0,955 | 0,967 | 0,973 | 0,975 | 0,977 |
| DiCE-genetic: cobertura | 0,057 | 0,056 | 0,056 | 0,056 | 0,056 |
| DiCE-random: cobertura, rodada original | **0,000** | **0,000** | **0,000** | 0,050 | 0,060 |
| DiCE-random: cobertura, após a correção do C9 | 0,052 | 0,056 | 0,057 | 0,059 | 0,060 |
| DiCE-random: sucesso, após a correção do C9 | 0,875 | 0,895 | 0,908 | 0,938 | 0,951 |
| Wachter: cobertura | 0,022 | 0,022 | 0,022 | 0,022 | 0,022 |
| FACE: cobertura | 0,021 | 0,021 | 0,021 | 0,021 | 0,021 |

**Comentário.**
- **O MAP-Elites não saturou em 10.000 consultas.** Passar de 5.000 para 10.000 consultas ainda acrescenta 8 células
  em média. No German Credit, a cobertura sobe de 0,12 para 0,34 ao longo da curva.
- **Os baselines são planos.** DiCE-genetic, Wachter e FACE convergem com pouquíssimas consultas e ignoram o
  orçamento extra. Isso confirma a observação do notebook 04 que motivou o desenho com curvas.
- **Na rodada original, o DiCE-random não devolvia nada com orçamentos ≤ 2.000.** Sua primeira chamada custava cerca
  de 2.031 consultas, e ele não tinha fallback anytime. Isso vinha do nosso adaptador, não do método (C9). Depois da
  correção, ele acha um contrafactual para 88% das instâncias com 500 consultas. A cobertura dele continua plana
  (0,05–0,06), então a vantagem do MAP-Elites não muda em nenhum orçamento.

### 2.4 Validade, sucesso e restrições

Fração das instâncias com pelo menos um contrafactual válido, com orçamento 10.000.

| Método | Adult | COMPAS | German Credit | GMC | HELOC |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAP-Elites | 1,000 | 0,947 | 1,000 | 0,957 | 0,982 |
| DiCE-genetic | 1,000 | 0,922 | 1,000 | 0,878 | 0,912 |
| DiCE-random | 1,000 | 0,857 | 1,000 | 0,955 | 0,902 |
| Wachter (LR, NN) | 1,000 | 0,940 | 1,000 | 0,985 | 1,000 |
| FACE | 0,845 | 0,900 | 0,882 | 0,873 | 0,777 |

**Comentário.**
- **As restrições valem em todo lugar:** 0 violações em 70.000 linhas, para todos os métodos, incluindo atributos
  imutáveis, só-aumenta e as regras de sentinela. É um resultado forte para os operadores de reparo compartilhados e
  para os adaptadores de DiCE, Wachter e FACE.
- **O Wachter tem o maior sucesso geral** (0,985, só LR e NN) e é o melhor em GMC e HELOC. O MAP-Elites deixa
  algumas instâncias sem solução em GMC, COMPAS e HELOC mesmo com 10.000 consultas, mas é o melhor no COMPAS.
- **O FACE é o mais fraco.** Ele devolve pontos reais dos dados, e projetá-los nos valores congelados da instância
  (contadores irreversíveis, idade) costuma perder a classe desejada. O notebook 03 trata isso como uma propriedade
  do FACE sob restrições de acionabilidade, não como bug.
- **O DiCE-genetic é fraco no GMC** (0,83 com LR, 0,81 com NN), onde o check contra o CARLA também deu números baixos
  (§2.8).

### 2.5 Proximidade e esparsidade

Para cada instância em que os dois métodos acharam um contrafactual válido, comparamos os contrafactuais mais
próximos de cada um (distância de Guidotti), com orçamento 10.000.

| MAP-Elites vs. | Instâncias pareadas | MAP-Elites mais próximo | Razão mediana (MAP-Elites ÷ baseline) |
| --- | ---: | ---: | ---: |
| DiCE-genetic | 2.826 | 99% | 0,29 |
| DiCE-random | 2.819 | 84% | 0,55 |
| Wachter | 1.950 | 99% | 0,24 |
| FACE | 2.565 | 99% | 0,17 |

Mediana da distância do mais próximo, por dataset:

| Método | Adult | COMPAS | German Credit | GMC | HELOC |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAP-Elites | **0,079** | **0,468** | **0,050** | **0,173** | **0,197** |
| DiCE-genetic | 0,423 | 0,667 | 0,239 | 0,462 | 0,536 |
| DiCE-random | 0,248 | 0,889 | **0,050** | 0,467 | 0,264 |
| Wachter | 0,389 | 1,198 | 0,260 | **53,7** | 0,423 |
| FACE | 0,536 | 1,222 | 0,475 | 0,680 | 0,795 |

**Comentário.**
- O método QD acha o contrafactual mais próximo, mesmo não sendo feito para minimizar distância. Duas coisas relativizam
  isso. O fitness do MAP-Elites *é* a distância de avaliação, `1 / (1 + distância de Guidotti)` (C2). E ele gasta
  10.000 consultas onde o DiCE-genetic para em cerca de 830 (C3).
- **Wachter no GMC (mediana 53,7, média 152 com LR)** é um problema de escala, não uma falha de busca. O GMC tem
  atributos de cauda pesada com MAD pequena. Um passo pequeno no espaço min-max do Wachter vira enorme em unidades de
  MAD (C7). Médias que incluem o GMC são dominadas por esses valores, por isso este relatório usa medianas.
- **Nenhum método é esparso o suficiente.** Cerca de dois terços dos atributos alterados pelo MAP-Elites, pelo
  DiCE-genetic e pelo FACE poderiam ser revertidos sem perder a classe desejada (C12).

### 2.6 Testes estatísticos

Testes de Friedman com um bloco por (dataset, modelo), com média sobre as seeds, e diferença crítica (CD) de Nemenyi
com α = 0,05. Rank menor é melhor.

**Sem o Wachter: 15 blocos, 4 métodos, CD = 1,21**

| Métrica | MAP-Elites | DiCE-random | DiCE-genetic | FACE | Friedman |
| --- | ---: | ---: | ---: | ---: | --- |
| Cobertura | **1,00** | 2,27 | 2,73 | 4,00 | χ² = 41,5, p = 5e-9 |
| Estratégias | **1,00** | 2,23 | 2,77 | 4,00 | χ² = 42,1, p = 4e-9 |
| Sucesso | **1,70** | 2,43 | 2,20 | 3,67 | χ² = 23,8, p = 3e-5 |
| Mediana da distância do mais próximo | **1,10** | 2,23 | 2,67 | 4,00 | χ² = 39,6, p = 1e-8 |

**Só LR e NN, com o Wachter: 10 blocos, 5 métodos, CD = 1,93**

| Métrica | MAP-Elites | DiCE-random | DiCE-genetic | Wachter | FACE | Friedman |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Cobertura | **1,00** | 2,10 | 2,90 | 4,00 | 5,00 | χ² = 39,3, p = 6e-8 |
| Sucesso | 2,30 | 2,75 | 3,05 | **2,10** | 4,80 | χ² = 25,4, p = 4e-5 |
| Mediana da distância do mais próximo | **1,10** | 2,30 | 3,00 | 4,00 | 4,60 | χ² = 30,9, p = 3e-6 |

**Comentário.**
- **Cobertura, 15 blocos:** o MAP-Elites difere significativamente de todos os baselines, mas por pouco do
  DiCE-random (diferença de rank 1,27 vs. CD 1,21).
- **Sucesso:** o MAP-Elites não difere significativamente de nenhuma das variantes do DiCE nem do Wachter.
- **Distância do mais próximo:** MAP-Elites vs. DiCE-random fica abaixo da CD (1,13 < 1,21).
- **Com o Wachter (10 blocos):** o Nemenyi **não** separa o MAP-Elites de nenhuma das variantes do DiCE na cobertura,
  embora o MAP-Elites vença em 10 de 10 blocos (teste do sinal bilateral, p = 0,002). Com tão poucos blocos, o
  Nemenyi é muito conservador (C13).
- Usar as seeds como blocos (150 blocos) dá p ≈ 1e-89. Isso é pseudo-replicação e não deve ser reportado.

### 2.7 Estudo de simulação (desenho completo, `results/simulation`)

Os dados sintéticos têm 4 rotas conhecidas para a classe favorável, cada uma formada por um par de atributos. A
rota 0 fica perto da instância típica, e as rotas 1–3 ficam longe. Na variante **alinhada**, os grupos do descritor
são as próprias rotas. Na variante **embaralhada**, os atributos das rotas são reagrupados ao acaso, então o grid não
coincide com as rotas. O desenho é 3 modelos × 4 métodos × 3 seeds × 10 instâncias × 3 orçamentos em cada variante:
66 configurações e 1.980 linhas, rodadas em 2026-11-21 com o código commitado. Substitui a versão rápida (1 seed,
5 instâncias), que mostrava o mesmo quadro.

| Método | Recall de rotas, alinhado (500 / 2.000 / 10.000) | Recall de rotas, embaralhado (500 / 2.000 / 10.000) | Rotas distantes achadas (de 3) em 10.000, alinhado / embaralhado | Execuções que acham as 4 rotas em 10.000, alinhado / embaralhado |
| --- | --- | --- | ---: | ---: |
| MAP-Elites | **0,69 / 0,67 / 0,66** | **0,66 / 0,61 / 0,59** | **1,68 / 1,47** | **17% / 8%** |
| DiCE-genetic | 0,34 / 0,34 / 0,34 | 0,34 / 0,34 / 0,34 | 0,42 / 0,42 | 0% / 0% |
| FACE | 0,19 / 0,19 / 0,19 | 0,19 / 0,19 / 0,19 | 0,26 / 0,26 | 0% / 0% |
| Wachter (LR, NN) | 0,07 / 0,07 / 0,07 | 0,07 / 0,07 / 0,07 | 0,12 / 0,12 | 0% / 0% |

**Comentário.**
- **O MAP-Elites acha cerca do dobro de rotas que o DiCE-genetic, nos 9 blocos (modelo, seed) das duas variantes.**
  Com 10.000 consultas, a vantagem sobre o melhor baseline é de +0,32 no alinhado e +0,26 no embaralhado, e nunca
  fica abaixo de +0,15.
- **A diferença vem das rotas distantes.** Os dois métodos acham a rota próxima (DiCE-genetic em 92% das instâncias,
  MAP-Elites em 94%), mas o MAP-Elites acha 1,68 das 3 rotas distantes, contra 0,42. Isso apoia a hipótese da
  simulação: os métodos clássicos ficam presos na rota mais próxima.
- **Embaralhar o grid custa mais do que na rodada rápida:** −0,07 em 10.000 (0,66 → 0,59), em vez de −0,03. O
  MAP-Elites continua bem à frente, então a simulação não é "feita sob medida", mas um grid alinhado ajuda. Os
  baselines não enxergam o grid, por isso seus números são iguais nas duas variantes, como esperado.
- **O recall cai quando o orçamento cresce** (0,69 → 0,66 no alinhado, 0,66 → 0,59 no embaralhado). Por instância, ele
  cai entre 2.000 e 10.000 consultas em 17% das execuções e sobe em 12%. Isso confirma o efeito de substituição de
  elites: um contrafactual mais próximo que não cumpre nenhuma rota verdadeira pode tirar do arquivo um elite que
  cumpria.
- **O MAP-Elites raramente acha todas as rotas:** as 4 em 17% das execuções (alinhado) e 8% (embaralhado).
- **A LR limita o recall,** porque não consegue representar o OU entre rotas: o MAP-Elites chega a 0,61 com LR vs.
  0,68 com NN e RF (alinhado, 10.000).
- **Muitos contrafactuais não cumprem nenhuma rota verdadeira.** Com 10.000 consultas, 42% dos contrafactuais válidos
  do MAP-Elites (33% no embaralhado) invertem o modelo sem satisfazer nenhuma rota, contra 19% do DiCE-genetic, 22% do
  FACE e 73% do Wachter. Eles exploram a diferença entre o modelo e a fronteira verdadeira (os dados têm 10% de ruído
  nos rótulos). Ver C18.
- O DiCE-genetic mostra 790 "violações" em cada variante. Todas são arredondamento float32 do atributo imutável (no
  máximo 1,2e-7) contra a tolerância de 1e-9 de `constraint_violations`. Não são mudanças reais (C19).

### 2.8 Check contra o CARLA (re-rodada, 200 instâncias por célula)

Sucesso com orçamento 10.000 na re-rodada: 10 seeds × 20 instâncias = 200 instâncias por (método, dataset, modelo),
no mesmo protocolo do resto do benchmark. Substitui o check anterior (notebook 03: 10 instâncias, seed 0). A
referência é a usada em `build_report_data.py`: 1,00 em tudo, exceto Wachter em Adult + NN (0,50) e em COMPAS + NN
(0,66). As células são LR / NN / RF.

| Método | Adult | COMPAS | German Credit | GMC | HELOC |
| --- | --- | --- | --- | --- | --- |
| MAP-Elites | 1,00 / 1,00 / 1,00 | 0,94 / 0,98 / 0,92 | 1,00 / 1,00 / 1,00 | 0,93 / 0,94 / 1,00 | 1,00 / 1,00 / 0,94 |
| DiCE-genetic | 1,00 / 1,00 / 1,00 | 0,91 / 0,94 / 0,92 | 1,00 / 1,00 / 1,00 | 0,83 / 0,81 / 1,00 | 1,00 / 0,92 / 0,81 |
| DiCE-random | 1,00 / 1,00 / 1,00 | 0,83 / 0,93 / 0,81 | 1,00 / 1,00 / 1,00 | 0,94 / 0,95 / 0,96 | 1,00 / 1,00 / 0,82 |
| Wachter (LR, NN) | 1,00 / 1,00 / – | 0,94 / 0,94 / – | 1,00 / 1,00 / – | 0,99 / 0,97 / – | 1,00 / 1,00 / – |
| FACE | 0,81 / 0,88 / 0,85 | 0,89 / 0,91 / 0,90 | 0,89 / 0,86 / 0,90 | 0,78 / 0,84 / 0,99 | 0,81 / 0,73 / 0,78 |

**Comentário.**
- **Os desvios em GMC + NN do check pequeno eram ruído.** Com 200 instâncias, o DiCE-genetic vai de 0,5 para 0,81, o
  FACE de 0,4 para 0,84 e o Wachter de 0,8 para 0,97.
- **O Wachter supera os números do CARLA em Adult + NN (1,00 vs. 0,50) e COMPAS + NN (0,94 vs. 0,66),** o que agora
  está confirmado com 200 instâncias. Como o plano aponta, o CARLA ignora restrições, usa seus próprios modelos e tem um
  timeout de 0,5 minuto.
- **O FACE fica abaixo da referência em todos os casos** (0,73–0,99). A causa é a mesma da §2.4: projetar pontos reais
  nos valores congelados da instância costuma perder a classe desejada.
- **O DiCE-genetic fica baixo no GMC (LR, NN) e em HELOC + RF** (0,81–0,83).
- 0 violações de restrição nas 14.000 linhas.

---

## 3. Situação das perguntas de pesquisa

- **RQ1 (QD vs. otimização):** a evidência aponta para "sim" para o MAP-Elites contra DiCE, Wachter e FACE. O
  MAP-Elites acha muito mais estratégias distintas, com a mesma validade e sem violar restrições, e seu contrafactual
  mais próximo é mais próximo. Faltam três coisas antes de afirmar isso: uma comparação com tamanho de conjunto
  igualado (C1), plausibilidade (C5) e o MOC (C6). O MOC devolve uma frente de Pareto, então é o baseline com mais
  chance de competir em cobertura.
- **RQ2 (QD vs. QD):** só pode ser estudada depois que NS, NSLC e surprise forem implementados (C6).

---

## 4. Preocupações

Em ordem de quanto ameaçam a conclusão principal. Cada preocupação traz a evidência e uma ação proposta.

### Altas: afetam a conclusão da RQ1

**C1. A comparação de cobertura é estruturalmente enviesada a favor do MAP-Elites.**
- *Evidência:* a cobertura conta células preenchidas, então é limitada pelo número de contrafactuais devolvidos. Wachter
  e FACE devolvem 1, o DiCE no máximo 10, e o MAP-Elites 85 em média (até 358). Em German Credit e HELOC, o teto do
  DiCE é 2,5% (§2.2). Por contrafactual devolvido, o DiCE é tão diverso em estratégias quanto o MAP-Elites, ou mais:
  German 0,67 estratégias por CF no DiCE-genetic vs. 0,38 no MAP-Elites; HELOC 0,44 vs. 0,30; GMC 0,41 vs. 0,30;
  Adult 0,28 vs. 0,25. O MAP-Elites também é o único método que enxerga o grid durante a busca.
- *Ação:* acrescentar uma comparação com **k igualado**, por exemplo o MAP-Elites restrito a k = 10 elites (melhor
  fitness, ou um subconjunto que maximize cobertura) vs. o DiCE com 10. Rodar o DiCE com `total_cfs` igual ao tamanho
  do arquivo do MAP-Elites. Reportar estratégias por contrafactual ao lado da cobertura. As linhas por contrafactual
  necessárias estão em `results/full/runs/` (Drive do grupo), não no git.

**C2. O MAP-Elites otimiza a métrica de avaliação.**
- *Evidência:* o fitness dos elites é `1 / (1 + distância de Guidotti)`, a mesma distância usada para avaliar
  proximidade. O Wachter minimiza L1 no espaço min-max, e o DiCE minimiza sua própria perda de proximidade +
  diversidade.
- *Ação:* reportar também a proximidade numa métrica neutra (por exemplo L1 no espaço min-max, ou a distância de cada
  método), e explicitar essa assimetria no artigo.

**C3. "Mesmo orçamento" não significa o mesmo esforço de busca.**
- *Evidência:* a mediana de consultas efetivamente usadas com orçamento 10.000 é 10.000 no MAP-Elites, cerca de 830
  no DiCE-genetic, 2.031 no DiCE-random, 9 no Wachter e 1 no FACE. Além disso, `train_predictions: free` dá ao DiCE
  (KD-tree) e ao FACE (grafo) predições não contadas sobre todo o treino (31 mil linhas no Adult, 84 mil no GMC).
- *Ação:* decidir em grupo a questão em aberto de `train_predictions`. Reportar as consultas usadas ao lado de toda
  curva. Considerar deixar o DiCE-genetic gastar o orçamento (mais gerações ou reinícios), para que as curvas comparem
  métodos que de fato tentam.

**C4. Os resultados misturam versões de código.**
- *Evidência:* o manifest tem 7 hashes de código-fonte diferentes. Só 262 configurações rodaram com o código
  commitado (`src:2144fd74c7`, quase todo o COMPAS e o GMC). As outras 438 rodaram com fontes mais antigas: todo o
  Adult, o German Credit e o HELOC, mais 18 configurações de COMPAS e GMC. As fontes antigas nunca foram commitadas,
  então não dá para mostrar o que mudou. Além disso, 25 entradas do manifest foram reconstruídas por
  `repair_manifest.py` (sem tempo, `reconstructed: true`).
- *Ação:* re-rodar as 438 configurações com o código commitado. São cerca de 23 horas de CPU, mais as 25 sem tempo
  registrado, algo como 4 h em 6 núcleos. Um primeiro passo mais barato é re-rodar uma amostra e comparar os
  summaries. Se baterem, documentar isso e manter os resultados.
- **Situação (2026-11-22): resolvida.** As 700 configurações foram re-rodadas com o código commitado, usando
  `scripts/rerun_pipeline.py`, em `results/full_rerun`. O `scripts/compare_runs.py` comparou as duas rodadas linha a
  linha em todas as colunas de resultado; a saída está em `results/full_rerun/compare_with_full.csv`.
  - **696 das 700 configurações são idênticas.** 20 das 60.764 linhas comparadas diferem. A conta deixa de fora as
    9.236 linhas do DiCE-random que a correção do C9 muda de propósito.
  - **As 4 configurações que diferem são todas do FACE, cada uma em uma única instância.** Em Adult + NN seed 3 e Adult
    + RF seed 7, o FACE devolve outro contrafactual mais próximo (por exemplo, distância 0,49 → 0,33). Em COMPAS + LR
    e COMPAS + NN seed 2, o contrafactual é o mesmo, mas o número de consultas muda.
  - **Todos os outros métodos são idênticos nos 7 hashes de código antigos,** então as mudanças de hash não mudaram
    nenhum resultado.
  - **As diferenças do FACE vêm de não-determinismo, não de mudança de código** (C20). Os resultados originais
    continuam valendo.

**C5. A plausibilidade não é medida.**
- *Evidência:* LOF, distância ao vizinho mais próximo e yNN não estão implementados (`metrics.py`), mas a RQ1 pede
  "sem perder plausibilidade". O MAP-Elites gera contrafactuais por mutação. Ele também produz valores fracionários
  em atributos inteiros: o notebook 03 mostra `duration: 48 → 34,05` meses, e o Wachter dá `48 → 38,29`.
- *Ação:* implementar as métricas de plausibilidade e a flag `integer` em `Feature` (arredondar em `clip`) **antes**
  da rodada final.

**C6. Metade dos métodos planejados está faltando.**
- *Evidência:* NS, NSLC, surprise e MOC não estão implementados, e o responsável pelo MOC continua em aberto. Sem eles
  não dá para estudar a RQ2, e a RQ1 fica sem seu baseline mais forte.
- *Ação:* definir o responsável pelo MOC. Implementar NS, NSLC e surprise via `register()`. O batch runner já dá
  suporte a eles.

### Médias: afetam a interpretação e a apresentação

**C7. As distâncias no GMC explodem.**
- *Evidência:* os atributos de cauda pesada do GMC (utilização, razão de dívida, renda) têm MAD muito pequena. A
  mediana da distância do mais próximo do Wachter no GMC é 53,7 (média 152 com LR), e a diversidade média entre pares
  do DiCE-random fica entre 273 e 302. Qualquer média que inclua o GMC é dominada por esses valores.
- *Ação:* reportar medianas, ou aplicar log/winsorização nos atributos de cauda pesada em `configs/gmc.yaml`.

**C8. Parte dos modelos explicados é fraca.**
- *Evidência:* os modelos do GMC mal superam a classe majoritária (0,935 vs. 0,931, acurácia balanceada 0,57–0,58) e
  preveem como desfavorável só cerca de 1,8% do teste. Os modelos do German Credit ficam em 0,71–0,73 contra uma
  maioria de 0,70, e a NN tem acurácia balanceada 0,60 (Guidotti reporta 0,81). Contrafactuais de classificadores
  próximos da classe majoritária dizem mais sobre fronteiras de decisão estranhas do que sobre os métodos.
- *Ação:* ajustar com pesos de classe ou um split de validação (nunca o teste). No mínimo, reportar a acurácia
  balanceada no artigo.

**C9. Os resultados do DiCE-random são em parte um artefato do adaptador, e ele supera o DiCE-genetic.**
- *Evidência:* o DiCE-random tem sucesso 0 com orçamentos ≤ 2.000 (§2.3). O DiCE-random supera o DiCE-genetic no
  rank médio de cobertura (2,27 vs. 2,73), na mediana da distância do mais próximo (0,26 vs. 0,41), na esparsidade
  (L0 1,7 vs. 4,4) e na redundância. O DiCE-genetic, nosso baseline principal, parece mal configurado.
- *Ação:* acrescentar um fallback anytime ao DiCE-random, ou amostragem que respeite o orçamento. Revisar os
  hiperparâmetros do DiCE-genetic antes que os revisores o façam.
- **Situação (2026-11-22): a parte do DiCE-random está corrigida** em `cfqd/methods/dice.py` (commit cde3eeb), com
  teste.
  - **O que mudou:** o tamanho da amostra agora acompanha o orçamento, `(orçamento − 1 − total_cfs) // 2`, limitado a
    1.000. Os candidatos válidos vistos durante a busca são guardados como fallback anytime.
  - **Orçamentos baixos:** na re-rodada, o sucesso com 500 / 1.000 / 2.000 consultas é 0,875 / 0,895 / 0,908 (era 0),
    e a cobertura fica entre 0,052 e 0,057. Ele nunca passa do orçamento e tem 0 violações.
  - **Quanto da saída é o fallback:** 87–91% das linhas com ≤ 2.000 consultas, 7% com 5.000 e 1% com 10.000. Nos
    orçamentos baixos, a saída são, portanto, as amostras aleatórias válidas, sem o pós-processamento do DiCE (como
    no C10).
  - **As linhas que a correção não deveria mexer não mudaram:** toda linha em que a rodada original terminou dentro do
    orçamento é idêntica.
  - **Orçamento 10.000:** o sucesso vai de 0,943 para 0,951 (HELOC 0,90 → 0,94), e a cobertura fica em 0,060.
  - **Ainda em aberto:** as configurações do DiCE-genetic.

**C10. O DiCE-genetic muitas vezes devolve o fallback, não sua resposta final.**
- *Evidência:* com orçamento 10.000, o fallback anytime dispara em 30% das instâncias (16–39% conforme o dataset).
  Com orçamento 500, dispara em quase todas as instâncias de Adult e German Credit. Essas saídas são os membros
  válidos da última população, sem o pós-processamento final do DiCE.
- *Ação:* reportar as taxas de fallback e separar os casos de fallback na análise.

**C11. O desenho do grid ainda é provisório.**
- *Evidência:*
  - O COMPAS tem 1 grupo, logo 2 estratégias e 10 células. Ele não consegue mostrar diversidade de estratégias.
  - Só 175 das 400 células do Adult são viáveis.
  - As faixas de esforço nunca foram calibradas antes da rodada completa (item em aberto no README).
  - Uma mudança num atributo nominal conta como "aumentou".
- *Ação:* calibrar as faixas de esforço num dataset. Decidir o estado para atributos nominais. Apresentar o COMPAS
  como caso de apêndice, como planejado.

**C12. A saída do MAP-Elites ainda não é uma explicação.**
- *Evidência:* 85 contrafactuais por instância em média (até 358), com 64% das mudanças reversíveis. Nenhum usuário vai
  ler isso.
- *Ação:* acrescentar uma etapa de seleção ou resumo (por exemplo, um elite por estratégia, esparsificação pós-hoc) e
  considerar um termo de esparsidade no fitness. Isso conversa com a questão em aberto sobre o fitness (plano, 5.3).

**C13. O desenho estatístico exige cuidado.**
- *Evidência:* usar seeds como blocos infla os p-valores em 80 ordens de grandeza. Os blocos corretos dão 10–15
  blocos, e com eles o Nemenyi não separa o MAP-Elites do DiCE apesar de 10/10 vitórias (§2.6).
- *Ação:* usar blocos (dataset, modelo). Acrescentar testes pareados de Wilcoxon ou do sinal com correção de Holm
  contra o método de controle. Mais datasets aumentam o poder mais do que mais seeds.

### Baixas: vale registrar

- **C14. O check contra o CARLA era pequeno.** Eram 10 instâncias e uma seed, e GMC + NN desviava para três métodos.
  *Resolvida em 2026-11-22:* a §2.8 agora usa 200 instâncias por célula da re-rodada, e os desvios em GMC + NN somem.
- **C15. A simulação era uma rodada rápida.** Uma seed e 5 instâncias. *Resolvida em 2026-11-22:* a §2.7 agora usa o
  desenho completo (3 seeds, 10 instâncias, 3 modelos, as duas variantes). O recall continua não monotônico no
  orçamento.
- **C16. As instâncias dependem do modelo.** Cada modelo explica as linhas que *ele* prevê como desfavoráveis, então
  comparações entre modelos não são pareadas. As comparações entre métodos são pareadas.
- **C17. As linhas por contrafactual não estão no git.** `results/full/runs/` (~200 MB) está no Drive, então qualquer
  reanálise no nível do contrafactual (C1, C12) precisa dessa cópia. O `results/full_rerun/runs/` da re-rodada também
  não está no git.

### Acrescentadas em 2026-11-22, após a re-rodada

**C18 (média). Muitos contrafactuais exploram erros do modelo.**
- *Evidência:* na simulação, com 10.000 consultas, 42% dos contrafactuais válidos do MAP-Elites (33% com o grid
  embaralhado) invertem o modelo sem satisfazer nenhuma rota verdadeira. A fração é de 19% no DiCE-genetic, 22% no
  FACE e 73% no Wachter (§2.7). O arquivo do MAP-Elites preenche células com esses pontos, e eles podem tirar do
  arquivo elites que estavam numa rota.
- *Ação:* reportar essa fração ao lado do recall de rotas. Medir plausibilidade (C5) e considerar um termo de suporte
  nos dados ou de plausibilidade no fitness.

- **C19 (baixa). As "violações" do DiCE na simulação são arredondamento.** O DiCE devolve valores em float32, então o
  atributo imutável contínuo se move em até 1,2e-7, e `constraint_violations` acusa isso com tolerância de 1e-9. Os
  datasets reais têm atributos imutáveis inteiros, por isso o problema não aparece neles. *Ação:* restaurar os valores
  congelados exatamente no adaptador do DiCE, ou usar tolerância de 1e-6.
- **C20 (baixa). O FACE não é totalmente determinístico.** Uma instância em cada uma de 4 configurações mudou entre
  rodadas idênticas (C4). A causa provável são empates no grafo kNN entre linhas duplicadas, desempatados de forma
  diferente entre as rodadas. *Ação:* acrescentar um desempate estável no grafo e na busca de caminho mínimo.

---

## 5. Próximos passos sugeridos

1. ~~Re-rodar, ou checar por amostragem, as 438 configurações que usaram código antigo (C4).~~ Feito em 2026-11-22:
   os resultados se reproduzem.
2. Fazer a análise de cobertura com k igualado e de estratégias por contrafactual (C1). Isso precisa da pasta `runs/`.
3. Implementar plausibilidade e a flag de inteiro (C5) e reportar a fração fora das rotas (C18). Calibrar as faixas de
   esforço (C11).
4. ~~Corrigir o fallback do DiCE-random~~ (feito, C9). Revisar as configurações do DiCE-genetic (C9, C10).
5. Decidir em grupo `train_predictions` e a escala do GMC (C3, C7).
6. Correções pequenas: o desempate do FACE (C20) e o arredondamento dos valores congelados no DiCE (C19).
7. Implementar MOC, NS, NSLC e surprise (C6), e depois rodar a grade final uma única vez, com uma única versão de
   código commitada.
