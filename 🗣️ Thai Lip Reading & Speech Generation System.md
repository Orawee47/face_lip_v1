# 🗣️ Thai Lip Reading & Speech Generation System

ระบบช่วยสื่อสารสำหรับผู้ที่ไม่สามารถเปล่งเสียงได้ โดยใช้เทคโนโลยี **Computer Vision, Lip Reading, Large Language Model (LLM) และ Thai Text-to-Speech (TTS)** เพื่อแปลงการเคลื่อนไหวของริมฝีปากเป็นข้อความภาษาไทย และแปลงข้อความดังกล่าวออกมาเป็นเสียงภาษาไทย

ระบบมีแนวคิดหลักคือ

```text
การขยับริมฝีปาก
       ↓
Lip Reading
       ↓
ข้อความภาษาไทย
       ↓
LLM
       ↓
ข้อความภาษาไทยที่ถูกต้องและเหมาะสมกับบริบท
       ↓
Thai TTS
       ↓
เสียงพูดภาษาไทย
```

---

# 📌 1. Project Overview

ผู้ที่มีภาวะหรือข้อจำกัดบางอย่างที่ทำให้ไม่สามารถเปล่งเสียงได้ อาจประสบปัญหาในการสื่อสารกับบุคคลรอบข้าง โดยเฉพาะการสื่อสารความต้องการพื้นฐาน เช่น

- ต้องการน้ำ
- ต้องการอาหาร
- ต้องการยา
- รู้สึกเจ็บ
- ต้องการความช่วยเหลือ
- ต้องการเรียกแพทย์หรือพยาบาล

โครงการนี้จึงมีแนวคิดในการสร้างระบบที่สามารถรับ **วิดีโอการขยับริมฝีปาก** แล้ววิเคราะห์เพื่อสร้างข้อความภาษาไทย และนำข้อความดังกล่าวไปสร้างเป็นเสียงพูด

ระบบประกอบด้วย 4 ส่วนสำคัญ:

1. **Computer Vision Pipeline**
2. **Thai Lip Reading Model**
3. **Large Language Model (LLM)**
4. **Thai Text-to-Speech (TTS)**

---

# 🎯 2. Project Objective

วัตถุประสงค์ของระบบคือ

- ตรวจจับและติดตามใบหน้าจากวิดีโอ
- ระบุตำแหน่งริมฝีปาก
- เตรียมข้อมูล Mouth Region สำหรับ Lip Reading
- สร้าง Dataset สำหรับภาษาไทย
- Train โมเดลสำหรับอ่านการเคลื่อนไหวของริมฝีปาก
- แปลงการเคลื่อนไหวของริมฝีปากเป็นข้อความภาษาไทย
- ใช้ LLM ช่วยแก้ไขข้อความที่มีความคลาดเคลื่อน
- ใช้บริบทของภาษาเพื่อช่วยเลือกคำที่เหมาะสม
- แปลงข้อความภาษาไทยเป็นเสียงพูด
- สร้างระบบที่สามารถนำไปต่อยอดเป็นเครื่องมือช่วยสื่อสารได้

---

# 🧠 3. System Concept

ระบบแบ่งการทำงานออกเป็น 3 แนวคิดหลัก

```text
SEE
 ↓
Lip Reading Model
 ↓
อ่านการเคลื่อนไหวของริมฝีปาก


UNDERSTAND
 ↓
LLM
 ↓
ตรวจสอบและประมวลผลข้อความ


SPEAK
 ↓
Thai TTS
 ↓
สร้างเสียงภาษาไทย
```

หรือสามารถสรุปเป็น

```text
VIDEO
  ↓
SEE
  ↓
UNDERSTAND
  ↓
SPEAK
  ↓
AUDIO
```

---

# 🏗️ 4. Overall System Architecture

```text
                         ┌───────────────┐
                         │  Video Input  │
                         └───────┬───────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Video Preparation    │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │  Face Detection        │
                    │  & Tracking            │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   Face Alignment       │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │      Face Crop          │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │      Mouth Crop         │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │    Clip Generation      │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │     Dataset & QA        │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ Thai Lip Reading Model │
                    │       Training          │
                    └───────────┬────────────┘
                                │
                                ▼
                         Trained Model
                                │
                                ▼
                         New Thai Video
                                │
                                ▼
                    ┌────────────────────────┐
                    │ Thai Lip Reading Model │
                    │      Prediction         │
                    └───────────┬────────────┘
                                │
                                ▼
                           Raw Thai Text
                                │
                                ▼
                    ┌────────────────────────┐
                    │          LLM           │
                    │ Correction & Context   │
                    └───────────┬────────────┘
                                │
                                ▼
                         Final Thai Text
                                │
                                ▼
                    ┌────────────────────────┐
                    │       Thai TTS         │
                    │    Text → Speech       │
                    └───────────┬────────────┘
                                │
                                ▼
                         🔊 Thai Audio
```

---

# 🔄 5. Complete Pipeline

ระบบแบ่งออกเป็น 12 Phase หลัก

```text
Phase 1
Video Preparation
        ↓
Phase 2
Face Detection & Tracking
        ↓
Phase 3
Face Alignment
        ↓
Phase 4
Face Crop
        ↓
Phase 5
Mouth Crop
        ↓
Phase 6
Clip Generation
        ↓
Phase 7
Dataset Builder & QA
        ↓
Phase 8
Lip Reading Model Training
        ↓
Phase 9
Lip Reading Prediction
        ↓
Phase 10
LLM Post-processing
        ↓
Phase 11
Thai Text-to-Speech
        ↓
Phase 12
Audio Output
```

---

# 📂 6. Phase 1 — Video Preparation

ขั้นตอนแรกคือการเตรียมวิดีโอให้อยู่ในรูปแบบที่เหมาะสมสำหรับระบบ

## งานหลัก

- ตรวจสอบไฟล์วิดีโอ
- ตรวจสอบ FPS
- ตรวจสอบ Resolution
- ตรวจสอบจำนวน Frames
- ตรวจสอบวิดีโอที่เสียหาย
- เตรียมข้อมูลสำหรับการประมวลผล

```text
Raw Video
    ↓
Video Validation
    ↓
Prepared Video
```

---

# 👤 7. Phase 2 — Face Detection & Tracking

ตรวจจับใบหน้าของผู้พูดในแต่ละ Frame และติดตามตำแหน่งใบหน้า

```text
Video Frame
      ↓
Face Detection
      ↓
Face Bounding Box
      ↓
Face Tracking
```

## Output

- Face Bounding Box
- Tracking ID
- Frame Information

ระบบสามารถใช้โมเดลตรวจจับใบหน้า เช่น YOLO ในขั้นตอนนี้

---

# 📐 8. Phase 3 — Face Alignment

หลังจากตรวจจับใบหน้าแล้ว จะทำการ Alignment เพื่อให้ตำแหน่งใบหน้ามีความสม่ำเสมอในแต่ละ Frame

ใช้ Facial Landmark เพื่อหาตำแหน่งสำคัญ เช่น

- ตา
- จมูก
- ปาก
- ขอบใบหน้า

```text
Face
 ↓
Facial Landmarks
 ↓
Calculate Transformation
 ↓
Aligned Face
```

เป้าหมายคือทำให้การเคลื่อนไหวของปากมีความสม่ำเสมอ และลดผลกระทบจากการเอียงหรือการเคลื่อนที่ของใบหน้า

---

# ✂️ 9. Phase 4 — Face Crop

ตัดเฉพาะบริเวณใบหน้าจาก Video Frame

```text
Full Frame
     ↓
Face Bounding Box
     ↓
Face Crop
```

ผลลัพธ์คือภาพใบหน้าที่พร้อมสำหรับการประมวลผลในขั้นตอนถัดไป

---

# 👄 10. Phase 5 — Mouth Crop

ตัดเฉพาะบริเวณริมฝีปากจากใบหน้าที่ผ่าน Alignment แล้ว

```text
Aligned Face
      ↓
Mouth Landmarks
      ↓
Mouth Bounding Box
      ↓
Mouth Crop
```

## Output

```text
Mouth Frames
```

ตัวอย่างโครงสร้างข้อมูล:

```text
mouth_frames/
├── video001/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   ├── frame_000003.jpg
│   └── ...
│
├── video002/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
```

Mouth Crop เป็นข้อมูลสำคัญสำหรับโมเดล Lip Reading เพราะโมเดลจะใช้การเปลี่ยนแปลงของบริเวณริมฝีปากเป็นข้อมูลหลักในการทำนาย

---

# 🎞️ 11. Phase 6 — Clip Generation

Lip Reading ไม่ได้วิเคราะห์เพียงภาพเดียว แต่ต้องวิเคราะห์ **Temporal Information**

ดังนั้นจึงต้องนำหลาย Frame มารวมกันเป็น Sequence หรือ Clip

```text
Frame 1
Frame 2
Frame 3
Frame 4
   .
   .
Frame N
   ↓
Temporal Sequence
   ↓
Lip Reading Clip
```

ตัวอย่าง:

```text
clip_001/
├── frame_001
├── frame_002
├── frame_003
├── ...
└── frame_029
```

ข้อมูลใน Clip จะเป็นตัวแทนของการเคลื่อนไหวของริมฝีปากในช่วงเวลาหนึ่ง

---

# 🗂️ 12. Phase 7 — Dataset Builder & Quality Assurance

หลังจากสร้าง Mouth Frames และ Clips แล้ว ต้องสร้าง Dataset สำหรับ Training

## ตรวจสอบข้อมูล

- จำนวน Frames
- จำนวน Clips
- Missing Frames
- Invalid Frames
- Sequence Length
- Labels
- Duplicate Data
- Train / Validation / Test Split

ตัวอย่าง:

```text
dataset/
├── train/
├── validation/
└── test/
```

---

# 🤖 13. Phase 8 — Thai Lip Reading Model Training

เมื่อ Dataset พร้อมแล้ว จึงเข้าสู่ขั้นตอนการ Train Lip Reading Model

แนวทางของโมเดลสามารถใช้สถาปัตยกรรม เช่น

- AV-HuBERT
- LipNet
- หรือโมเดล Lip Reading ที่เหมาะสมกับ Dataset

แม้ AV-HuBERT จะรองรับข้อมูลภาพและเสียงในงานดั้งเดิม แต่สำหรับระบบนี้สามารถพิจารณาการใช้งานเฉพาะ Visual Information ตามรูปแบบการทดลองที่กำหนด

## Training Flow

```text
Thai Mouth Clips
       ↓
Preprocessing
       ↓
Feature Extraction
       ↓
Lip Reading Model
       ↓
Loss Calculation
       ↓
Backpropagation
       ↓
Model Update
```

ทำซ้ำหลาย Epoch จนโมเดลสามารถเรียนรู้ความสัมพันธ์ระหว่าง

```text
Lip Movement
      ↓
Thai Words / Thai Sentences
```

## Output

```text
trained_model/
├── model
├── config
└── checkpoint
```

---

# 🔮 14. Phase 9 — Thai Lip Reading Prediction

หลังจาก Train โมเดลแล้ว สามารถนำ Video ใหม่เข้าสู่ระบบ

```text
New Thai Video
       ↓
Face Detection
       ↓
Face Alignment
       ↓
Mouth Crop
       ↓
Clip Generation
       ↓
Trained Lip Reading Model
       ↓
Prediction
```

ตัวอย่าง:

```text
Video
 ↓
Lip Reading Model
 ↓
"ฉันต้องการนํา"
```

ผลลัพธ์นี้เรียกว่า

**Raw Prediction**

ซึ่งอาจมีความคลาดเคลื่อน เช่น

```text
"ฉันต้องการนํา"
```

แทนที่จะเป็น

```text
"ฉันต้องการน้ำ"
```

สาเหตุอาจเกิดจากความคล้ายกันของการเคลื่อนไหวริมฝีปาก หรือข้อจำกัดของโมเดลในการแยกคำที่มีรูปปากใกล้เคียงกัน

---

# 🧠 15. Phase 10 — LLM Post-processing

หลังจาก Lip Reading Model ได้ข้อความแล้ว จึงส่งข้อความเข้าสู่ LLM

**LLM ไม่ได้ทำหน้าที่อ่านปาก**

หน้าที่ของ LLM คือประมวลผลข้อความที่ได้จาก Lip Reading Model

```text
Raw Thai Text
      ↓
     LLM
      ↓
Context Analysis
      ↓
Word Correction
      ↓
Sentence Correction
      ↓
Final Thai Text
```

## ตัวอย่าง

```text
Raw Prediction:

"ฉันต้องการนํา"
```

LLM วิเคราะห์บริบท:

```text
"ฉันต้องการน้ำ"
```

จากนั้นส่งข้อความที่ผ่านการปรับแล้วไปยัง TTS

---

# ⚠️ 16. LLM Safety & Correction Rules

LLM ควรได้รับคำสั่งให้รักษาความหมายเดิมของข้อความ

หลักการ:

```text
1. แก้เฉพาะคำที่มีแนวโน้มผิด
2. รักษาความหมายเดิม
3. ไม่เพิ่มข้อมูลที่ไม่มีใน Input
4. ไม่ลบข้อมูลสำคัญ
5. ไม่สร้างประโยคใหม่โดยไม่มีเหตุผล
6. พิจารณาบริบททางการแพทย์
```

ตัวอย่าง:

```text
Input:
"ฉันต้องการยา"

ควรได้:
"ฉันต้องการยา"

ไม่ควรเปลี่ยนเป็น:
"ฉันต้องการน้ำ"
```

แม้ว่าคำว่า "น้ำ" จะเป็นคำที่พบได้บ่อยกว่า

---

# 🔊 17. Phase 11 — Thai Text-to-Speech

เมื่อได้ Final Thai Text แล้ว จะเข้าสู่ขั้นตอน Text-to-Speech

TTS มีหน้าที่แปลง

```text
Thai Text
    ↓
Thai Speech
```

ตัวอย่าง:

```text
"ฉันต้องการน้ำ"
       ↓
    Thai TTS
       ↓
🔊 เสียงภาษาไทย
```

---

# 🗣️ 18. Thai TTS Options

สามารถพิจารณา TTS ได้ 2 แนวทาง

## Cloud TTS

ตัวอย่าง:

- Google Cloud Text-to-Speech
- Microsoft Azure Speech

### ข้อดี

- รองรับภาษาไทย
- คุณภาพเสียงดี
- ใช้งานง่าย
- ไม่ต้องดูแลโมเดล TTS เอง

### ข้อจำกัด

- ต้องใช้อินเทอร์เน็ต
- อาจมีค่าใช้บริการ
- ข้อมูลข้อความถูกส่งไปยังบริการภายนอก

---

## Local / Open-source TTS

ใช้โมเดลที่สามารถรันบนเครื่องหรือ Server ของระบบ

```text
Thai Text
    ↓
Local TTS Model
    ↓
Audio
```

### ข้อดี

- สามารถทำงาน Offline ได้
- ควบคุมข้อมูลได้มากกว่า
- ลดการพึ่งพา Cloud

### ข้อจำกัด

- ต้องเลือกโมเดลภาษาไทยที่เหมาะสม
- ต้องใช้ทรัพยากรเครื่อง
- คุณภาพเสียงแตกต่างกันตามโมเดล

---

# 🔊 19. Phase 12 — Audio Output

หลังจาก TTS สร้าง Audio แล้ว ระบบสามารถนำเสียงไปเล่นผ่าน Speaker

```text
Final Thai Text
       ↓
Thai TTS
       ↓
Audio
       ↓
Speaker
       ↓
🔊 Thai Speech
```

ตัวอย่าง Output:

```text
output/
├── prediction.txt
├── corrected_text.txt
└── speech.wav
```

---

# 🔁 20. End-to-End Example

สมมติผู้ป่วยต้องการพูดว่า

```text
"ฉันต้องการน้ำ"
```

ระบบจะทำงานดังนี้

### Step 1 — Video

```text
🎥 Video
```

ผู้ป่วยขยับริมฝีปากเพื่อสื่อสาร

### Step 2 — Lip Reading

```text
Video
 ↓
Lip Reading Model
 ↓
"ฉันต้องการนํา"
```

### Step 3 — LLM

```text
"ฉันต้องการนํา"
        ↓
       LLM
        ↓
"ฉันต้องการน้ำ"
```

### Step 4 — TTS

```text
"ฉันต้องการน้ำ"
        ↓
      Thai TTS
        ↓
     🔊 Audio
```

### Step 5 — Speaker

ผู้ฟังจะได้ยินเสียงภาษาไทย:

```text
"ฉันต้องการน้ำ"
```

---

# 🧪 21. Training Pipeline

ส่วน Training จะมีเฉพาะกระบวนการที่เกี่ยวข้องกับ Lip Reading Model

```text
Thai Video Dataset
       ↓
Video Preparation
       ↓
Face Detection
       ↓
Face Alignment
       ↓
Face Crop
       ↓
Mouth Crop
       ↓
Clip Generation
       ↓
Dataset Builder
       ↓
Dataset QA
       ↓
Train / Validation / Test
       ↓
Thai Lip Reading Model
       ↓
Training
       ↓
Trained Model
```

### LLM และ TTS ไม่จำเป็นต้องอยู่ใน Training Pipeline หลัก

เหตุผลคือ

- Lip Reading Model เรียนรู้ความสัมพันธ์ระหว่าง Lip Movement กับ Text
- LLM ทำหน้าที่ Post-processing
- TTS ทำหน้าที่แปลง Text เป็น Audio

ดังนั้นแต่ละส่วนสามารถพัฒนาและประเมินแยกกันได้

---

# 🚀 22. Inference Pipeline

เมื่อมี Trained Model แล้ว ระบบจริงจะทำงานดังนี้

```text
New Thai Video
       ↓
Video Preprocessing
       ↓
Face Detection
       ↓
Face Alignment
       ↓
Mouth Crop
       ↓
Clip Generation
       ↓
Thai Lip Reading Model
       ↓
Raw Thai Text
       ↓
LLM
       ↓
Final Thai Text
       ↓
Thai TTS
       ↓
Thai Audio
       ↓
Speaker
```

---

# 📊 23. Model Evaluation

ระบบควรประเมินแยกตามแต่ละส่วน

## 23.1 Lip Reading Model

สามารถใช้ Metrics เช่น

- Word Error Rate (WER)
- Character Error Rate (CER)
- Sentence Accuracy
- Word Accuracy

ตัวอย่าง:

```text
Ground Truth:
ฉันต้องการน้ำ

Prediction:
ฉันต้องการนํา
```

นำ Prediction และ Ground Truth มาเปรียบเทียบเพื่อคำนวณ Error Rate

---

# 📈 24. LLM Evaluation

ควรเปรียบเทียบผลลัพธ์ก่อนและหลังใช้ LLM

```text
Before LLM
     ↓
Raw Prediction
```

เปรียบเทียบกับ

```text
After LLM
     ↓
Corrected Prediction
```

ตัวอย่าง:

```text
Ground Truth:
ฉันต้องการน้ำ

Raw:
ฉันต้องการนํา

After LLM:
ฉันต้องการน้ำ
```

จากนั้นสามารถเปรียบเทียบว่า LLM ช่วยลด Error ได้มากน้อยเพียงใด

---

# 🔊 25. TTS Evaluation

สำหรับ TTS สามารถประเมินด้านต่าง ๆ เช่น

- ความชัดเจน
- ความเป็นธรรมชาติ
- ความถูกต้องของการออกเสียง
- ความเร็วในการสร้างเสียง
- Latency
- ความสามารถในการทำงาน Offline
- Resource Usage

หากมีหลาย TTS Model สามารถทดลองเปรียบเทียบกันได้

---

# 📊 26. Overall Evaluation

ระบบสามารถประเมินเป็น 3 Stage

```text
Stage 1
Lip Reading
Video → Text

        ↓

Stage 2
LLM
Raw Text → Correct Text

        ↓

Stage 3
TTS
Text → Audio
```

ทำให้สามารถระบุได้ว่าปัญหาเกิดขึ้นที่ส่วนใดของระบบ

---

# 🗂️ 27. Suggested Project Structure

โครงสร้างโปรเจกต์สามารถจัดได้ดังนี้

```text
LipReading/
│
├── data/
│   ├── raw/
│   ├── prepared/
│   ├── aligned/
│   ├── mouth_frames/
│   ├── clips/
│   └── dataset/
│
├── notebooks/
│   ├── phase1_video_preparation.ipynb
│   ├── phase2_face_detection_tracking.ipynb
│   ├── phase3_face_alignment.ipynb
│   ├── phase4_face_crop.ipynb
│   ├── phase5_mouth_crop.ipynb
│   ├── phase6_clip_generation.ipynb
│   ├── phase7_dataset_builder.ipynb
│   ├── phase7.5_dataset_QA.ipynb
│   ├── phase8_model_training.ipynb
│   ├── phase9_prediction.ipynb
│   ├── phase10_llm.ipynb
│   └── phase11_tts.ipynb
│
├── models/
│   ├── face_detection/
│   ├── lip_reading/
│   └── tts/
│
├── src/
│   ├── preprocessing/
│   ├── face_detection/
│   ├── face_alignment/
│   ├── mouth_crop/
│   ├── clip_generation/
│   ├── dataset/
│   ├── lip_reading/
│   ├── llm/
│   └── tts/
│
├── outputs/
│   ├── predictions/
│   ├── corrected_text/
│   └── audio/
│
├── configs/
│
├── requirements.txt
│
└── README.md
```

---

# 🛠️ 28. Technologies

เทคโนโลยีหลักที่สามารถใช้ในระบบประกอบด้วย

### Computer Vision

- Python
- OpenCV
- YOLO
- MediaPipe
- Facial Landmark

### Deep Learning

- PyTorch
- CUDA
- Lip Reading Model
- AV-HuBERT / LipNet

### Language Model

- Large Language Model (LLM)
- API-based LLM หรือ Local LLM

### Speech

- Thai Text-to-Speech
- Cloud TTS หรือ Local TTS

### Development

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

# 🔐 29. Privacy Considerations

เนื่องจากระบบเกี่ยวข้องกับวิดีโอของผู้ป่วย ควรให้ความสำคัญกับความเป็นส่วนตัวของข้อมูล

ควรพิจารณา:

- ไม่เผยแพร่วิดีโอผู้ป่วยโดยไม่ได้รับอนุญาต
- ไม่ Commit ข้อมูลส่วนบุคคลขึ้น GitHub
- ไม่เก็บข้อมูลที่ไม่จำเป็น
- จำกัดสิทธิ์การเข้าถึง Dataset
- หากใช้ Cloud LLM หรือ Cloud TTS ต้องพิจารณาการส่งข้อมูลออกนอกระบบ
- ควรลบหรือปกปิดข้อมูลที่สามารถระบุตัวบุคคลได้เมื่อไม่จำเป็น

---

# ⚙️ 30. Installation

สร้าง Virtual Environment:

```bash
python -m venv lipreading_env
```

เปิดใช้งานบน Windows:

```bash
lipreading_env\Scripts\activate
```

ติดตั้ง Dependencies:

```bash
pip install -r requirements.txt
```

ตรวจสอบ PyTorch:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
```

---

# ▶️ 31. Running the Pipeline

เริ่มจากการเตรียม Video:

```text
Phase 1
↓
Phase 2
↓
Phase 3
↓
Phase 4
↓
Phase 5
↓
Phase 6
↓
Phase 7
```

หลังจาก Dataset ผ่าน QA แล้วจึง Train:

```text
Phase 8
```

หลังจากได้ Trained Model:

```text
Phase 9
↓
Phase 10
↓
Phase 11
↓
Phase 12
```

---

# 🧩 32. Modular Design

ระบบถูกออกแบบให้แต่ละ Component สามารถเปลี่ยนแยกกันได้

ตัวอย่างเช่น

```text
Lip Reading Model
        ↓
       Text
        ↓
      LLM A
        ↓
       Text
        ↓
     TTS A
```

สามารถเปลี่ยนเป็น

```text
Lip Reading Model
        ↓
       Text
        ↓
      LLM B
        ↓
       Text
        ↓
     TTS B
```

โดยไม่จำเป็นต้องเปลี่ยน Pipeline ทั้งหมด

---

# 🔮 33. Future Development

## 33.1 Improve Thai Lip Reading

เพิ่ม Dataset ภาษาไทยเพื่อให้โมเดลสามารถรองรับคำศัพท์ได้มากขึ้น

```text
More Thai Data
      ↓
Better Training
      ↓
Better Lip Reading
```

---

## 33.2 Medical Vocabulary

เพิ่มคำศัพท์ที่เกี่ยวข้องกับการสื่อสารของผู้ป่วย เช่น

```text
น้ำ
ยา
เจ็บ
ปวด
หายใจ
ช่วยด้วย
หมอ
พยาบาล
โรงพยาบาล
```

---

## 33.3 Context-aware Prediction

สามารถใช้บริบทก่อนหน้าเพื่อช่วยให้ LLM เลือกคำได้ดีขึ้น

```text
Previous Sentence
        +
Current Raw Prediction
        ↓
       LLM
        ↓
Context-aware Text
```

---

## 33.4 Real-time Lip Reading

ในอนาคตสามารถพัฒนาจากการประมวลผล Video แบบ Offline ไปเป็น Real-time

```text
Camera
  ↓
Real-time Face Detection
  ↓
Mouth Tracking
  ↓
Lip Reading
  ↓
LLM
  ↓
Thai TTS
  ↓
🔊
```

---

# 📌 34. Important Design Principle

ระบบนี้ไม่ได้ใช้ LLM เพื่ออ่านริมฝีปากโดยตรง

แต่แบ่งหน้าที่ดังนี้:

```text
┌────────────────────────────┐
│ Lip Reading Model          │
│                            │
│ Lip Movement → Thai Text   │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ LLM                        │
│                            │
│ Raw Text → Better Text     │
└─────────────┬──────────────┘
              ↓
┌────────────────────────────┐
│ Thai TTS                   │
│                            │
│ Thai Text → Thai Speech    │
└────────────────────────────┘
```

ดังนั้น

> **Lip Reading Model = อ่านปาก**

> **LLM = ประมวลผลภาษา**

> **Thai TTS = สร้างเสียง**

---

# 🏁 35. Final System

เมื่อพัฒนาเสร็จ ระบบจะสามารถทำงานตาม Flow:

```text
                 🎥 VIDEO
                     │
                     ▼
             Face Detection
                     │
                     ▼
              Face Tracking
                     │
                     ▼
              Face Alignment
                     │
                     ▼
                Face Crop
                     │
                     ▼
                Mouth Crop
                     │
                     ▼
             Clip Generation
                     │
                     ▼
          Thai Lip Reading Model
                     │
                     ▼
              Thai Raw Text
                     │
                     ▼
                   LLM
                     │
                     ▼
            Final Thai Text
                     │
                     ▼
                 Thai TTS
                     │
                     ▼
              🔊 Thai Audio
                     │
                     ▼
                 Speaker
```

## Core Pipeline

```text
VIDEO
  ↓
LIP READING
  ↓
THAI TEXT
  ↓
LLM
  ↓
FINAL THAI TEXT
  ↓
THAI TTS
  ↓
THAI SPEECH
```

ระบบมีเป้าหมายในการเปลี่ยน **การสื่อสารผ่านการขยับริมฝีปาก** ให้กลายเป็น **ข้อความและเสียงภาษาไทย** เพื่อช่วยให้ผู้ที่มีข้อจำกัดในการเปล่งเสียงสามารถสื่อสารกับบุคคลรอบข้างได้สะดวกมากขึ้น