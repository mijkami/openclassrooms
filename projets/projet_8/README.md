# étapes installation projet

# installation prérequis
# java
sudo pacman -S jdk17-openjdk
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
java -version
echo $JAVA_HOME

# aws-cli
sudo pacman -S aws-cli
aws configure

> https://us-east-1.console.aws.amazon.com/iam/home?region=eu-west-3#/users
> aller dans P8_userclient
> Security Credentials
> Access keys /create acces key

> aws s3api s3://p8-data-dgdev

aws s3 mb s3://p8-data-dgdev
aws s3 ls
cd data/Test1
aws s3 sync . s3://p8-data-dgdev/Test

> upload bootstrap

aws s3 cp bootstrap-emr.sh s3://p8-data-dgdev/

> upload .pem

zip "P8_aws_key.zip"  "P8_aws_key.pem"

aws s3 cp "P8_aws_key.zip" "s3://p8-data-dgdev/certificates/P8_AWS_key.zip"






# activation instance EC2 sur AWS

# création utilisateur sur AWS + droits S3

# connection SSH avec fichier .pem
chmod 600 P8_aws_key.pem
make aws_connect


# cluster EC2 Software Settings JSON
[
  {
    "Classification": "jupyter-s3-conf",
    "Properties": {
      "s3.persistence.enabled": "true",
      "s3.persistence.bucket": "p8-data-dgdev"
    }
  }
]


