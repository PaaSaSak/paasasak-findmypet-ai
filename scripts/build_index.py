import json, os, glob, numpy as np
from PIL import Image
import onnxruntime as ort, faiss

IMG_SIZE=224
serv="serving"; os.makedirs(serv, exist_ok=True)

sess = ort.InferenceSession("serving/embed.onnx", providers=["CUDAExecutionProvider","CPUExecutionProvider"])

def embed_image(path):
    img = Image.open(path).convert('RGB').resize((IMG_SIZE,IMG_SIZE))
    x = np.asarray(img).astype('float32')/255.0
    x = np.transpose(x,(2,0,1))[None,...]
    emb = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    faiss.normalize_L2(emb)
    return emb

paths=[]; ids=[]; metas={}
for pid in sorted(os.listdir("dataset/embed/val")):
    pdir = os.path.join("dataset/embed/val", pid)
    if not os.path.isdir(pdir): continue
    for p in glob.glob(os.path.join(pdir,"*.jpg")):
        paths.append(p); ids.append(int(pid))
        metas[str(len(paths)-1)]={"pid": int(pid), "path": p}

embs=[embed_image(p)[0] for p in paths]
embs=np.stack(embs).astype('float32')
faiss.normalize_L2(embs)

index = faiss.index_factory(512,"HNSW32,IDMap")
index.hnsw.efConstruction = 200
index.add_with_ids(embs, np.arange(len(embs)).astype('int64'))
faiss.write_index(index, "serving/index.faiss")
json.dump({"id_to_meta": metas}, open("serving/meta.json","w"), ensure_ascii=False)
print("saved serving/index.faiss & serving/meta.json")
