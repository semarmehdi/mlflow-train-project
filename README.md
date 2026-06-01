# MLflow Project — Training & Deployment Guide

## 1. Test en local (OBLIGATOIRE avant EC2)

### En local

```bash
git clone https://github.com/semarmehdi/mlflow-train-project.git
cd mlflow-train-project
```

```bash
conda env create -f conda.yaml
```

à partir du conda.yaml suivant :

```yaml
name: mlflow-train
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - mlflow==3.5.0
      - scikit-learn==1.6.1
      - pandas>=2.0
      - numpy>=1.26
      - psycopg2-binary
      - jupyter
      - openpyxl
      - requests>=2.31.0,<3
      - boto3
      - gunicorn
      - matplotlib
      - seaborn
      - plotly
```

```bash
conda activate mlflow-train
```

---

## 2. Build et test Docker en local

### En local

```bash
docker build . -t mlflow3-mehdi
docker tag mlflow3-mehdi mehdisemar2/mlflow3-mehdi
docker push mehdisemar2/mlflow3-mehdi
```

```bash
docker run mehdisemar2/mlflow3-mehdi
```

---

```bash
source .env
```

```bash
mlflow run . --experiment-name california_housing_regressor
```

# ATTENTION :

-on part d'une instance EC2 t3small éligible au freetier (vous pouvez gardez tout les parametres par défaut et simplement rajouter plus de stockage tout en bas - jusqu'à 30gb - )

## 3. Setup EC2

Pour rentrer dans EC2 :

```bash
ssh -i ssh_key/keydsfs40.pem ec2-user@ec2-15-188-63-7.eu-west-3.compute.amazonaws.com
```

### Dans EC2

```bash
sudo yum update -y
sudo yum install docker -y
```

```bash
echo -e "sudo service docker start" >> .bashrc
sudo usermod -a -G docker ec2-user
```

```bash
sudo yum install git -y
sudo yum install python3 -y
sudo yum install python3-pip -y
```

---

## 4. Installer Python 3.11 avec pyenv

### Dans EC2

```bash
sudo yum groupinstall "Development Tools" -y
sudo yum install gcc zlib-devel bzip2 bzip2-devel readline-devel sqlite sqlite-devel openssl-devel xz-devel libffi-devel -y
```

```bash
curl https://pyenv.run | bash
```

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

```bash
source ~/.bashrc
```

```bash
pyenv install 3.11.8
pyenv global 3.11.8
```

```bash
python --version
```

```bash
pip install --upgrade pip
pip install mlflow==3.5.0
```
### ATTENTION
Fermez maintenant votre instance dans votre terminal avec `exit` et relancez la pour que toutes les configurations soient prises en compte. 

---

## 5. Gestion des secrets
*Dans un autre terminal* 
### En local

Créer un fichier `secrets.sh` :

```bash
export MLFLOW_TRACKING_URI=...
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export ARTIFACT_ROOT=...
export BACKEND_STORE_URI=...
```

### En local (envoi vers EC2)

```bash
scp -i ssh_key/keydsfs40.pem secrets.sh ec2-user@ec2-15-188-63-7.eu-west-3.compute.amazonaws.com:~/
```

### Dans EC2

```bash
source secrets.sh
```

---

## 6. Lancer le projet MLflow sur EC2

### Dans EC2

```bash
mlflow run VOTRE_REPO_GIT --experiment-name california_housing_regressor
```

---

## Bonnes pratiques

- Toujours tester en local avant cloud
- Dockeriser pour reproductibilité
- Ne jamais push les credentials
