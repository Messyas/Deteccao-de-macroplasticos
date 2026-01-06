# Detecção de Macroplásticos através da Aplicação de Ensemble de Modelos de Aprendizado Profundo

Repositório com código e scripts para reproduzir os experimentos do artigo **"Detecção de macroplásticos através da aplicação de ensemble de modelos de aprendizado profundo"**, apresentado no Congresso Brasileiro de Inteligência Computacional (CBIC 2025). O estudo aplica Redes Neurais Convolucionais (ex.: YOLOv11x) e a técnica de fusão *Weighted Boxes Fusion* (WBF) para melhorar a detecção de resíduos plásticos em ambientes naturais.

**Artigo:** [SBIA/CBIC 2025 - CBIC2025-1191737](https://sbia.org.br/eventos/cbic_2025/cbic2025-1191737/)

## Ambiente de reprodução

Configure um ambiente Conda com suporte a GPU e as principais bibliotecas usadas nos experimentos.

```bash
git clone git@github.com:Messyas/Deteccao-de-macroplasticos-igarapes.git
cd Deteccao-de-macroplasticos-igarapes

conda create -n plastico_detect python=3.9
conda activate plastico_detect

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ultralytics
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install opencv-python matplotlib pandas jupyterlab
```

- Ajuste a versão do PyTorch/CUDA conforme sua GPU.
- O Detectron2 deve corresponder à versão instalada do PyTorch.

## Datasets

- **Treinamento:** dataset anotado do trabalho original (arXiv: [2401.14719](https://arxiv.org/abs/2401.14719)).
- **Teste (Igarapés de Manaus):** conjunto não anotado para inferência e validação visual (Kaggle - Igarapés de Manaus).

Estrutura sugerida para organizar os dados:

```text
datasets/
  train/
  test/
```

## Como usar

1. Ative o ambiente `plastico_detect`.
2. Baixe/posicione os datasets na estrutura acima.
3. Ajuste os caminhos de entrada/saída nos scripts em `Modelos/`, `utils/` e `visualizations/` conforme sua máquina.
4. Execute os scripts de treinamento ou inferência e avalie as métricas/saídas geradas.

## Citação

Se este código ou os datasets forem úteis na sua pesquisa, cite:

França, M. G., & Machado, M. C. (2025). Detecção de macroplásticos através da aplicação de ensemble de modelos de aprendizado profundo. In *Anais do XVII Congresso Brasileiro de Inteligência Computacional*. SBIA. DOI: 10.21528/CBIC2025-1191737.
