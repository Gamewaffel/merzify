import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, AutoencoderKL
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
import insightface

from transformers import CLIPVisionModelWithProjection
from diffusers.models import ControlNetModel

import os
import sys

# InstantID lokal importieren (nach Clone)
sys.path.insert(0, './InstantID')
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

class MerzifyPipeline:
    def __init__(self):
        self.pipe = None
        self.face_app = None
        self.merz_embeds = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Nutze: {self.device}")

    def load_models(self, merz_image_path: str):
        print("⏳ Lade Gesichtserkennung...")
        # InsightFace für Gesichtserkennung
        self.face_app = FaceAnalysis(
            name='antelopev2',
            root='models',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        print("⏳ Lade ControlNet...")
        controlnet_path = "models/instantid/ControlNetModel"
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16
        )

        print("⏳ Lade Basis-Modell (SDXL)...")
        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # Besser: RealVisXL für realistische Gesichter
        # base_model = "SG161222/RealVisXL_V4.0"

        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # IP-Adapter laden
        self.pipe.load_ip_adapter_instantid('models/instantid/ip-adapter.bin')
        self.pipe.to(self.device)

        print("⏳ Extrahiere Merz Gesichts-Embeddings...")
        self.merz_embeds, self.merz_kps = self._get_face_embeds(merz_image_path)
        print("✅ Pipeline bereit!")

    def _get_face_embeds(self, image_path: str):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.face_app.get(img)
        if not faces:
            raise ValueError(f"Kein Gesicht in {image_path} gefunden!")
        face = faces[0]
        embeds = torch.from_numpy(face.normed_embedding).unsqueeze(0)
        kps = face.kps
        return embeds, kps

    def merzify(
        self,
        prompt: str = "a photo of a person, highly detailed, realistic",
        negative_prompt: str = "blurry, low quality, distorted, ugly",
        num_steps: int = 30,
        guidance_scale: float = 5.0,
        ip_adapter_scale: float = 0.8,
        controlnet_scale: float = 0.8,
        seed: int = 42,
        width: int = 1024,
        height: int = 1024,
    ) -> Image.Image:
        
        if self.pipe is None:
            raise RuntimeError("Pipeline nicht geladen! Rufe load_models() auf.")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        # Gesichts-Keypoints als ControlNet Input
        face_kps_image = self._draw_kps(
            np.zeros((height, width, 3), dtype=np.uint8),
            self.merz_kps * np.array([width/640, height/640])
        )

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=self.merz_embeds.to(self.device, dtype=torch.float16),
            image=face_kps_image,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
        ).images[0]

        return image

    def _draw_kps(self, img, kps, color_list=[(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255)]):
        stickwidth = 4
        limbSeq = np.array([[0,2],[1,2],[3,2],[4,2]])
        kps = np.array(kps)
        for i in range(len(kps)):
            x, y = kps[i]
            color = color_list[i % len(color_list)]
            cv2.circle(img, (int(x), int(y)), 10, color, -1)
        for i, limb in enumerate(limbSeq):
            cur_canvas = img.copy()
            Y = kps[limb, 0]
            X = kps[limb, 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0]-X[1])**2 + (Y[0]-Y[1])**2)**0.5
            angle = np.degrees(np.arctan2(X[0]-X[1], Y[0]-Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), 
                (int(length/2), stickwidth), 
                int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(cur_canvas, polygon, color_list[i % len(color_list)])
            img = cv2.addWeighted(img, 0.4, cur_canvas, 0.6, 0)
        return Image.fromarray(img)
