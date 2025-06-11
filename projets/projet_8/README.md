# étapes installation projet

# installation prérequis

sudo pacman -S jdk17-openjdk
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
java -version
echo $JAVA_HOME
sudo pacman -S aws-cli


# activation instance EC2 sur AWS

# création utilisateur sur AWS + droits S3

# connection SSH avec fichier .pem
chmod 600 P8_aws_key.pem
make aws_connect
