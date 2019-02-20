# Hosting jupyter notebooks over the server

### Easier method over ssh tunnerl
This is a very easy way to host jupyter notebooks over a server if we only want to use it temporarily. First on the server side, start the jupyter notebook on a port
```
jupyter notebook --no-browser --port=4444
```
Then on the client side, make a ssh tunnel to access this port
```
ssh -N -f -L localhost:1234:localhost:4444 <server ip or name>
```
In the above example, the workbook can be access in
```
localhost:1234
```

### Permanent running in the background
To do this, we first need to make some config files. 
- Copy the `jupyter_notebook_config.py` file in this repo to the following location on server: `~/.jupyter/jupyter_notebook_config.py`. Make the directory `~/.jupyter/` if it doesn't exist.
- Run the following command: `jupyter notebook password`. This should allow for password setup and will make the following file: `~/.jupyter/jupyter_notebook_config.json`
- Run this command from to generate a `.pem` file and `.key` for secure connection to the server. This is required by jupyter to work reliably. 
```
cd ~/.jupyter/
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem
```
- Edit the `~/.jupyter/jupyter_notebook_config.py` file as follows: it should have some lines that are commented out --- only those are relevent. Look for the line `c.NotebookApp.password = 'sha1:<.....>'`. You need to replace this line with the sha1 hash in the file `~/.jupyter/jupyter_notebook_config.json`. 
- The other relevent line is `c.NotebookApp.port = 4444` where you can setup the port that you want to use.
- Make a screen session so that the command is always running in the background and host the notebook.
```
cd ~
screen
jupyter notebook --certfile ~/.jupyter/mycert.pem --keyfile ~/.jupyter/mykey.key
```
- The workbook can be accessed at:
```
https://<server>:4444
```