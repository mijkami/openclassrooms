# étapes installation projet

# docs :
https://docs.aws.amazon.com/
https://awsforengineers.com/blog/aws-ec2-and-s3-for-beginners/
https://repost.aws/
https://docs.aws.amazon.com/fr_fr/AWSEC2/latest/UserGuide/AmazonS3.html
https://jupyter-enterprise-gateway.readthedocs.io/en/v2.6.0/
https://alysivji.github.io/setting-up-pyenv-virtualenvwrapper.html
https://www.youtube.com/playlist?list=PL7iMyoQPMtAN4xl6oWzafqJebfay7K8KP (AWS Course)


# local linux (/!\ current distro : archlinux, check other package managers)
## java
> create your P8_kernel to contain the packages installed for the project
> link P8_kernel to jupyter to be able to select it

sudo pacman -S jdk17-openjdk
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
java -version
echo $JAVA_HOME

## aws-cli : upload files
sudo pacman -S aws-cli
aws configure

> https://us-east-1.console.aws.amazon.com/iam/home?region=eu-west-3#/users
> aller dans P8_userclient
> Security Credentials
> Access keys /create acces key

aws s3api s3://p8-data-dgdev

aws s3 mb s3://p8-data-dgdev
aws s3 ls
cd data/Test1
aws s3 sync . s3://p8-data-dgdev/Test

## makefile + .env infos
> create .env file in root to be able to use makefile
> add .env file to .gitignore for security reasons
> .env file content : 
AWS_SSH_HOST= amazonaws-host-adress-to-copy-from-cluster-page
AWS_SSH_USER= hadoop-or-username-from-cluster-page
AWS_KEY= your-key-pair-name-used-in-cluster-creation.pem

AWS_access_key_id= YourAccessID
AWS_secret_acces_key= YourAccessKey

> connect SSH to check : 
make aws_connect
> hit "yes" => connected to main node


### contenu bootstrap : cf file
sudo python3 -m pip install -U setuptools
sudo python3 -m pip install -U pip
sudo python3 -m pip install wheel
sudo python3 -m pip install pillow
sudo python3 -m pip install pandas==1.2.5
sudo python3 -m pip install pyarrow==4.0.1
sudo python3 -m pip install boto3==1.26.0
sudo python3 -m pip install s3fs==2023.1.0
sudo python3 -m pip install fsspec==2023.1.0
sudo python3 -m pip install namex
sudo python3 -m pip install rich
sudo python3 -m pip install dm-tree

### upload bootstrap
aws s3 cp bootstrap-emr.sh s3://p8-data-dgdev/bootstrap-emr.sh


# activation instance EC2 sur AWS
> Amazon EMR > EMR on EC2:Clusters > Create Cluster
> Name : P8_Spark_cluster
> EMR release : last / 7.9.0
> Application bundle : Tensorflow / JupyterHub / Spark
> Size / other : minimum
> Networking : blank / auto
> Cluster termmination : 1 hour
> Bootstrap actions : Amazon S3 location : S3://you-bucket/bootstrap-emr.sh
> Software settings > enter configuration
[
  {
    "Classification": "jupyter-s3-conf",
    "Properties": {
      "s3.persistence.bucket": "p8-data-dgdev",
      "s3.persistence.enabled": "true"
    }
  }
]

> Security configuration and EC2 key pair
  > Security configuration : none
  > Amazon EC2 key pair for SSH : add/select key pair
> IAM roles : blank / auto
  > Amzon EMR service role : don't change
  > EC2 instance profile for Amazon EMR : edit to add S3 access
> Validate "Create/Clone cluster"


# Page cluster (ex:P8_spark_cluster)
> laisser tourner install + bootstrap
> vérifier Status pour infos sur install
> Cluster management > Primary node public DNS : copy into .env to use make aws_connect in local linuX CLI for ssh connect
> bottom left : Network and Security > show details > EMR managed security group link 
  > edit inbound rules : add two rules (/!\ full access for ease of use, can be a security liability)
    > ALL UDP 0.0.0.0:0
    > ALL UDP 0.0
    > Save rules

## connection SSH avec fichier .pem
> Cluster management > Primary node public DNS : copy into .env
chmod 400 P8_aws_key.pem
make aws_connect

## FoxyProxy (Firefox)
> install FoxyProxy 
> add to toolbar > open Options
> Add/Ajouter :
  > name : EMR
  > Type : SOCKS5
  > Hostname : localhost
  > Port: 5555
  > Proxy DNS : checked
  > rest : blank/empty
  > Save
> Select EMR proxy in Firefox toolbar

## Jupyter Hub
> get JupyterHub link on EC2 cluster page
  > tab Appliactions
  > JupyterHub link : click

> upload prod notebook to JupyterHub user
aws s3 cp Prod_Pyspark.ipynb s3://your-s3-bucket/jupyter/jovyan/Prod_pyspark.ipynb

> JupyterHub via SSH
  > login : jovyan
  > pwd   : jupyter

> run cells
> check results in your S3 bucket /Results folder


# Shutdown :
> Terminate tous les clusters/serveurs créés
> vérifier alertes / recap budget