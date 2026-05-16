# aws cloud deployment

```
whoami
```

## sudo apt-get update

sudo → admin/root permission
apt-get → package manager
update → package indexes refresh

```
sudo apt-get update
```

## install docker

```
sudo apt-get install -y docker.io
```

- install → package install
- -y → automatic yes

## add admin in docker group me user

```
sudo usermod -aG docker $USER
```

- usermod → user modify
- -aG → append to group
- docker → docker group
- $USER → current user

```
docker login
```

## check docker user command :

```
sudo usermod -aG docker $USER
```

## use docker

```
newgrp docker
```

## check docker version

```
docker --version
```

```
docker run hello-world
```

## docker services check command

```
sudo systemctl status docker
sudo systemctl start docker
sudo systemctl enable docker
```

## pull image

```
docker pull sharjeelahmed017/traffic-insights

docker images

docker run -d -p 3000:3000 --name traffic-app sharjeelahmed017/traffic-insights

docker ps


docker logs traffic-app

docker logs -f traffic-app

docker stop traffic-app


docker rm traffic-app

docker start traffic-app
```

## production build command

```
docker run -d -p 80:3000 --restart unless-stopped --name traffic-app sharjeelahmed017/traffic-insights
```

# add security rule

tcp + port 5173 + allow anywhere

- give public ip address + your port
