### Build
`image_name` is free as docker image name．<br>
```
sudo docker build -t image_name .
cd ..
```

### Run Container
`$pwd` is mount current dir．<br>
```
sudo docker run -it --shm-size 2g --gpus all -v $(pwd):/workspace image_name 
```
