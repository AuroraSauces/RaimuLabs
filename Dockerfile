FROM continuumio/miniconda3:latest

WORKDIR /app

RUN conda create -n myenv python=3.12 -y

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install everything from conda-forge only
RUN conda install -c conda-forge -y \
    numpy=2.0.2 \
    scikit-learn=1.6.1 \
    dlib=19.24.6 \
    opencv=4.12.0 \
    pillow \
    flask && conda clean -afy

COPY . .

EXPOSE 7860

CMD ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "app.py"]
