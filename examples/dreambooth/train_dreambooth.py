from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from googleapiclient import discovery
import os
import pathlib
import paramiko
from paramiko.buffered_pipe import BufferedPipe, PipeTimeout
global service
import tarfile
import time
import traceback
service = None
project = 'affable-bastion-369921'
zone = 'us-central1-f'
instanceName = 'instance-1'
dreamboothRoot = "/home/tessacoilvr_gmail_com/dreambooth"
codeRoot = '/home/tessacoilvr_gmail_com/diffusers'
# server.run("ps aux | grep python | grep tessaco+ | grep [t]rain_dreambooth")
class RemoteServer(object):
  def __init__(self):
    self.ssh = None
    
    
  def testPython(self):
    #sudo /opt/deeplearning/install-driver.sh
    #pip3 uninstall torch
    #pip3 install torch
    #pip install torch --extra-index-url https://download.pytorch.org/whl/cu112
    res = self.run("/opt/conda/bin/python3.7 /home/tessacoilvr_gmail_com/testPython.py")
    if res.strip() != 'torch.Size([4, 5])':
      print("torch failed, making symlinks")
      print(res)
      '''
      for f in ['libcublas.so.11', 'libcublasLt.so.11', 'libnvblas.so.11']:
        p = '/opt/conda/lib/python3.7/site-packages/nvidia/cublas/lib/' + f
        linkPath = '/opt/conda/pkgs/cudatoolkit-11.1.1-h6406543_10/lib/' + f
        self.run("sudo rm " + p)
        self.run("sudo ln -s " + linkPath + " " + p, debug=True)
      '''
    
  def connect(self):
    while True:
      didConnect, state = initInstance()
      time.sleep(2.0)
      print(didConnect, state)
      if didConnect:
        break
    self.tryStartup()
    
  def tryStartup(self):
    success, state = initInstance()
    if success and self.ssh is None:
      externalIp = getInstance()['networkInterfaces'][0]['accessConfigs'][0]['natIP']
      self.externalIp = externalIp
      self.ssh = paramiko.SSHClient()
      self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
      self.ssh.connect(externalIp, username="tessacoilvr_gmail_com", password='', key_filename="privtessh")
      print("connected ssh to ip " + externalIp)
      self.sftp = self.ssh.open_sftp()
      print("connected sftp")
      stdin, stdout, stderr = self.ssh.exec_command('ls')
      print(stdout.readlines())
    else:
      print(success, state)
  
  def run(self, command, debug=False, streaming=False):
    if debug: print(command)
    if streaming:
      channel = self.ssh.invoke_shell()
      prompt = (command + "\n").encode()
      channel.send(prompt)
      accumulated = []
      promptStr = 'tessacoilvr_gmail_com@instance-1:~$ '.encode()
      firstTime = True
      while True:
        try:
          h = channel.in_buffer.read(1, timeout=1)
        except PipeTimeout:
          continue
        accumulated += [h]
        if b"".join(accumulated) == promptStr[:len(accumulated)]:
          if b"".join(accumulated) == promptStr:
            # it echos back what we typed, filter that and everything before it out
            if firstTime:
              firstTime = False
              # ignore prompt
              channel.recv(len(prompt)+1)
              accumulated = []
            else:
              break
        else:
          if len(accumulated) > 0:
            if not firstTime:
              print(b"".join(accumulated).decode("ascii", "ignore"), end="")
          accumulated = []
      channel.close()
    else:
      stdin, stdout, stderr = self.ssh.exec_command(command)
      allRes = ""
      for l in stdout.readlines():
        print(l, end='')
        allRes += l
      for l in stderr.readlines():
        print(l, end='')
      return allRes
  
  def upload(self, localPath, remotePath, debug=False):
    if debug: print("upload", localPath, "to", remotePath)
    self.sftp.put(localPath, remotePath)
  
  def download(self, remotePath, localPath, debug=False):
    if debug: print("download", remotePath, "to", localPath)
    self.sftp.get(remotePath, localPath)
    
  def uploadFolder(self, localFolder, remoteFolder):
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(localFolder)) for f in fn]
    dirPath = pathlib.Path(localFolder).resolve()
    localFolderParts = dirPath.parts
    targets = []
    dirs = set([str(pathlib.Path(remoteFolder)).replace("\\", "/")])
    for f in files:
      fullPath = pathlib.Path(f).resolve()
      fullPathParts = fullPath.parts
      partsAfterLocalFolder = fullPathParts[len(localFolderParts):]
      localPath = str(fullPath).replace("\\", "/")
      outFullPath = pathlib.Path(remoteFolder, pathlib.Path(*partsAfterLocalFolder))
      outFullPathDir = pathlib.Path(*outFullPath.parts[:-1])
      dirs.add(str(outFullPathDir).replace("\\", "/"))
      outPath = str(outFullPath).replace("\\", "/")
      targets.append((localPath, outPath))
    # make any needed intermediate directories
    for d in dirs:
      self.run("mkdir -p " + str(d))
      print("mkdir -p " + str(d))
    for localPath, outPath in targets:
      print(localPath, outPath)
      self.upload(localPath, outPath)
  def tryDisconnect(self):
    if not self.ssh is None:
      if not self.sftp is None:
        self.sftp.close()
        self.sftp = None
        print("disconnect sftp")
      self.ssh.close()
      self.ssh = None
      print("disconnect ssh")
  
  def tryShutdown(self):
    self.tryDisconnect()
    success, state = stopInstance()
    print(success, state)
      
      
  def setupDreambooth(self, datasetFolder, projectTag):
    # clone diffusers
    self.run("rm -rf " + codeRoot, debug=True)
    self.run("git clone https://github.com/Phylliida/diffusers", debug=True)
    
    # create project directories (and remove any existing ones)
    projectFolder = dreamboothRoot + "/" + projectTag
    imagesFolder = projectFolder + "/" + "images"
    self.run("mkdir -p " + projectFolder, debug=True)
    self.run("rm -rf " + imagesFolder, debug=True)
    
    # tar the images is faster than one file at a time
    tarfilePath = projectTag + ".tar.gz"
    tar = tarfile.open(tarfilePath, "w:gz")
    tar.add(datasetFolder, arcname="images")
    tar.close()
    outTarfilePath = projectFolder + "/images.tar.gz"
    self.upload(tarfilePath, outTarfilePath, debug=True)
    self.run("tar -xvzf " + outTarfilePath + " -C " + projectFolder, debug=True)
    self.run("rm " + outTarfilePath, debug=True)
    os.remove(tarfilePath)
    
  def runDreambooth(self, projectTag, prompt, saveNSteps):
    while True:
      try:
        self.connect()
        f = open("hugging face key.txt", "r")
        hfKey = f.read().strip()
        f.close()
        projectFolder = dreamboothRoot + "/" + projectTag
        imagesFolder = projectFolder + "/" + "images"
        existingModels = self.run("ls " + projectFolder)
        models = []
        startStep = 0
        for m in existingModels.split("\n"):
          end = "_unet.pkl"
          if m[-len(end):] == end:
            numSteps = int(m.split("_")[1])
            models.append((numSteps, m))
        resumeUnet = ""
        if len(models) > 0:
          models.sort(key=lambda x: -x[0])
          startStep = models[0][0]
          resumeUnet = '--unet_resume="' + projectFolder + "/" + models[0][1] + '"'
          print("resuming with unet", models[0][1], 'with',models[0][0],'steps')
        command = '/opt/conda/bin/python3.7 diffusers/examples/dreambooth/train_dreambooth.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --instance_data_dir="' + imagesFolder + '" --output_dir="' + projectFolder + '" --learning_rate=2e-6 --lr_scheduler="polynomial" --use_8bit_adam --max_train_steps=70000 --gradient_accumulation_steps=1 --mixed_precision="fp16" --save_starting_step=' + str(startStep) + ' --Session_dir="' + projectFolder + '" --instance_prompt="' + prompt + '" --seed=27 --resolution=512 --train_batch_size=1 --hub_token="' + hfKey + '"' + ' --save_n_steps=' + str(saveNSteps) + ' ' + resumeUnet + ' ' + '--run_tag="' + projectTag + '" --image_captions_filename'
        print(command)
        self.run(command, streaming=True)
      except Exception as e:
        print(traceback.print_exc())
        print("tasks got exception", e)
        if type(e) is KeyboardInterrupt:
            raise e
  

def getService():
  global service
  if service is None:
    service = discovery.build('compute', 'v1')
  return service
  
def getInstance():
  service = getService()
  requestList = service.instances().list(project=project, zone=zone)
  response = requestList.execute()
  for instance in response['items']:
    if instance['name'] == instanceName:
      return instance
  raise Exception("could not find instance", instanceName)
  
def instanceStatus():
  return getInstance()['status']
  
def initInstance():
  service = getService()
  status = instanceStatus()
  if status in ['STOPPING']:
    return False, status
  if status in ['STAGING', 'PROVISIONING', 'REPAIRING']:
    return False, status
  if status in ['TERMINATED']:
    requestStart = service.instances().start(project=project, zone=zone, instance=instanceName)
    response = requestStart.execute()
    return False, status
  if status in ['RUNNING']:
    return True, status
  raise Exception("Unknown status " + status)
    
def stopInstance():
  service = getService()
  status = instanceStatus()
  if status in ['STOPPING']:
    return False, status
  if status in ['STAGING', 'PROVISIONING', 'REPAIRING']:
    return False, status
  if status in ['RUNNING']:
    requestStop = service.instances().stop(project=project, zone=zone, instance=instanceName)
    response = requestStop.execute()
    return False, status
  if status in ['TERMINATED']:
    return True, status
  raise Exception("Unknown status " + status)


class GDriveFetcher(object):
  def __init__(self, tag, unettag):
    self.tag = tag
    self.unettag = unettag
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    self.drive = GoogleDrive(gauth)
    rootFolder = '1-1W9J5NmOmZAWRTb08E3Ja_NVxwf2gWf'
    self.modelPathFolder = self.getSubfolder(rootFolder, tag)['id']
  
  def getSubfolder(folderId, subfolderName):
    query = "'{}' in parents and trashed=false"
    query=query.format(rootFolder)
    file_list = self.drive.ListFile({'q': query}).GetList()
    print([x['title'] for x in file_list])
    return [x for x in file_list if x['title'] == tag][0]
    
  def fetch(self):
    query = "'{}' in parents and trashed=false"
    query=query.format(self.modelPathFolder)
    file_list = self.drive.ListFile({'q': query}).GetList()
    models = []
    for f in file_list:
      if f['title'][:len(self.tag)] == tag:
        t, st, stpunet = f['title'].split("_")
        step = int(stpunet[:-len("unet.pkl")])
        print("found model with", step, "steps")
        models.append((step, f))
    # get most recent model
    if len(models) > 0:
      resSteps, res = models.sort(key=lambda x: -x[0])[0]
      print("most recent model:", resSteps, res['title'])
      outPath = "pklmodels/unet-" + self.unettag + ".pkl"
      if os.path.exists(outPath):
        os.remove(outPath)
      print("downloading to", outPath)
      res.GetContentFile(outPath)
    
