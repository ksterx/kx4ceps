FROM continuumio/miniconda3

LABEL maintainer="ksterx <kostonerx@gmail.com>"

ENV HOME /home
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    curl \
    git \
    build-essential \
    vim \
    sudo
# RUN curl -sL https://deb.nodesource.com/setup_13.x | sudo -E zsh –
RUN apt-get install -y \

    # tkinter requirements
    python3-tk \
    libgl1-mesa-dev

# Visual Studio Code Live Share requirements
# libssl1.1 \
# libkrb5-3 \
# zlib1g \
# libicu[0-9][0-9] \
# gnome-keyring \
# libsecret-1-0 \
# desktop-file-utils \
# x11-utils &&\
# apt-get clean && \
# rm -rf /var/lib/apt/lists/*

ARG USERNAME=ksterx
ARG PASSWORD=pwd
RUN useradd -m -s /bin/bash -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    gpasswd -a $USERNAME sudo

######################
# Package installation (conda)

RUN conda update --all -y -q && \
    conda install -y -q \
    pillow \
    seaborn \
    typing
RUN conda install pytorch torchvision torchaudio -c pytorch
RUN conda install -c conda-forge nodejs==15.14.0
RUN conda clean --all -f -y

# Package installation (pip)
RUN pip install -U pip && \
    pip install \
    bokeh \
    kedro \
    kedro-viz \
    line-profiler \
    mlflow \
    opencv-python \
    opencv-contrib-python \
    optuna \
    pytorch_memlab \
    torch-summary

######################
# Preferences (Comment the lines out, if not needed)

RUN pip install -q \
    black \
    jupyterlab-kite>=2.0.2 \
    jupyterlab-git \
    jupyter-lsp \
    jupyter-tensorboard \
    jupytext \
    nbresuse \
    'python-language-server[all]' \
    ipympl

RUN conda install -y -q \
    ipywidgets \
    && conda install -y -q xeus-python -c conda-forge

######################
# Jupyter setup
RUN jupyter notebook --generate-config \
    && echo "c.NotebookApp.ip = '*'" >> $HOME/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.port = 8888" >> $HOME/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.open_browser = False" >> $HOME/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.notebook_dir = '$WORKSPACE'" >> $HOME/.jupyter/jupyter_notebook_config.py \
    && echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"' >> $HOME/.jupyter/jupyter_notebook_config.py \
    && echo 'c.ContentsManager.default_jupytext_formats = "ipynb,py"' >> $HOME/.jupyter/jupyter_notebook_config.py


######################
# JupyterLab Extensions

# RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
# RUN jupyter labextension install @lckr/jupyterlab_variableinspector
# RUN jupyter labextension install @krassowski/jupyterlab-lsp
# RUN jupyter labextension install @jupyterlab/debugger
# RUN jupyter labextension install @telamonian/theme-darcula
# RUN jupyter labextension install @jupyterlab/toc
# RUN jupyter labextension install jupyterlab_tensorboard
# RUN jupyter labextension install jupyterlab-topbar-extension jupyterlab-system-monitor
# RUN jupyter labextension install @kiteco/jupyterlab-kite
# RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib@0.7.4
# RUN jupyter labextension install @bokeh/jupyter_bokeh

# RUN jupyter lab build

WORKDIR $USERNAME/workspace