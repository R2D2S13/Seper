query = 'Where is the football?'
image_path = 'data/image/1.jpg'
from configs import config
config.multiprocessing = False
from utils.key_utils import KeyManager
km = KeyManager()
km.start()
from prog_env_V2 import ProgWrapper
pw = ProgWrapper(
)
pw.file_name = 'test'
c,cs = pw.generate_code(query=query)
varpool = pw.initiative_varpool(image_path)
res = pw.run_program(cs,varpool)
print(f'res:{res}')