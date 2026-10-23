import io, time
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

st.set_page_config(page_title="DefenceVision AI", page_icon="🛰️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:#08111f;color:#edf4ff}.block-container{max-width:1450px;padding-top:1.2rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1729,#101f35);border-right:1px solid #1c3554}
[data-testid="stSidebar"] *{color:#e9f2ff!important}
.hero{background:linear-gradient(120deg,#102642,#12385a 55%,#18304c);border:1px solid #275176;border-radius:22px;padding:28px 30px;margin-bottom:18px}
.kicker{color:#62d5ff;font-size:11px;font-weight:800;letter-spacing:2px}
.hero h1{font-size:clamp(2rem,4vw,3.2rem);margin:10px 0;color:#f5f9ff;font-weight:800}
.hero p{color:#b6c9df;max-width:800px}
.panel{background:#0e1b2e;border:1px solid #203954;border-radius:18px;padding:19px}
.section-title{color:#f0f6ff;font-size:1.12rem;font-weight:800;margin-bottom:5px}
.muted{color:#91a8c3;font-size:.85rem}
.metric{background:linear-gradient(145deg,#0f2035,#12263e);border:1px solid #254361;border-radius:15px;padding:16px;min-height:90px}
.metric-label{color:#91abc6;font-size:11px;text-transform:uppercase;letter-spacing:1px;font-weight:700}
.metric-value{color:#f4f8ff;font-size:1.45rem;font-weight:800;margin-top:7px}
div.stButton>button{background:linear-gradient(90deg,#087fbd,#2563eb);color:white;border:0;border-radius:11px;min-height:44px;font-weight:700}
hr{border-color:#203954}footer{visibility:hidden}
</style>
""",unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO("yolov8n.pt")

def font_for(size):
    try: return ImageFont.truetype("arial.ttf",size)
    except Exception: return ImageFont.load_default()

def annotate(image,result):
    out=image.copy().convert("RGB"); draw=ImageDraw.Draw(out); font=font_for(max(14,out.width//65)); detections=[]
    for box in result.boxes:
        cls_id=int(box.cls[0].item()); conf=float(box.conf[0].item())
        x1,y1,x2,y2=[int(v) for v in box.xyxy[0].tolist()]
        label=str(result.names.get(cls_id,cls_id))
        detections.append({"Object":label,"Confidence (%)":round(conf*100,1),"X1":x1,"Y1":y1,"X2":x2,"Y2":y2})
        color=(35,205,255); draw.rectangle((x1,y1,x2,y2),outline=color,width=max(3,out.width//350))
        text=f"{label} {conf:.0%}"
        try:
            bb=draw.textbbox((x1,max(0,y1-26)),text,font=font)
            draw.rectangle((bb[0]-4,bb[1]-3,bb[2]+4,bb[3]+3),fill=(7,25,43))
        except Exception: pass
        draw.text((x1,max(0,y1-26)),text,fill=color,font=font)
    return out,detections

with st.sidebar:
    st.markdown("## 🛰️ DefenceVision **AI**")
    st.caption("COMPUTER VISION LAB")
    st.markdown("---")
    page=st.radio("NAVIGATION",["🧭 Detection Lab","📊 Analytics","📘 Project Details"])
    st.markdown("---")
    st.markdown("### SYSTEM STATUS")
    st.markdown("**Model**  \nYOLOv8 nano\n\n**Mode**  \nUploaded-image analysis\n\n**Runtime**  \nLocal machine")
    st.caption("Educational prototype. No live surveillance or tracking.")

st.markdown("""<div class="hero"><div class="kicker">✦ AI-POWERED COMPUTER VISION</div>
<h1>DefenceVision AI</h1><p><b>Intelligent Object Detection System</b><br>
Analyze uploaded images with a pretrained object-detection model. View object locations, class labels, confidence scores and detection summaries in one workspace.</p></div>""",unsafe_allow_html=True)
st.info("Educational demonstration only. Use images you have permission to analyze. This prototype does not provide identity recognition, live tracking, or operational defence guidance.")

if page=="🧭 Detection Lab":
    left,right=st.columns([.85,1.15],gap="large")
    with left:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown('<div class="section-title">01 / Upload image</div><div class="muted">JPG, JPEG, PNG or WEBP</div><br>',unsafe_allow_html=True)
        uploaded=st.file_uploader("Choose an image",type=["jpg","jpeg","png","webp"])
        conf=st.slider("Minimum confidence",.10,.90,.35,.05)
        st.caption("Higher thresholds usually show fewer detections.")
        run=st.button("⚡ Run Object Detection",use_container_width=True,disabled=uploaded is None)
        if uploaded:
            original=Image.open(uploaded).convert("RGB")
            st.image(original,caption="Uploaded image",use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown('<div class="section-title">02 / Detection output</div><div class="muted">Bounding boxes and labels appear here.</div><br>',unsafe_allow_html=True)
        if run and uploaded:
            try:
                with st.spinner("Loading AI model and analyzing image… first run may download model weights."):
                    model=load_model(); source=Image.open(uploaded).convert("RGB"); start=time.time()
                    result=model.predict(source=source,conf=conf,verbose=False)[0]
                    annotated,detections=annotate(source,result); elapsed=time.time()-start
                st.session_state["dv_annotated"]=annotated; st.session_state["dv_detections"]=detections
                st.session_state["dv_elapsed"]=elapsed
            except Exception as e:
                st.error("Could not run the model. Check internet for first-time model download, then retry.")
                st.code(str(e))
        if "dv_annotated" in st.session_state:
            annotated=st.session_state["dv_annotated"]; detections=st.session_state["dv_detections"]
            st.image(annotated,caption="AI annotated output",use_container_width=True)
            a,b,c=st.columns(3); a.metric("Objects found",len(detections)); b.metric("Inference time",f'{st.session_state.get("dv_elapsed",0):.2f}s'); c.metric("Status","Processed")
            buff=io.BytesIO(); annotated.save(buff,format="PNG")
            st.download_button("⬇️ Download annotated image",buff.getvalue(),"defencevision_annotated.png","image/png",use_container_width=True)
            if detections: st.dataframe(pd.DataFrame(detections)[["Object","Confidence (%)"]],use_container_width=True,hide_index=True)
            else: st.warning("No objects detected above this threshold. Try lowering it or using a clearer image.")
        else:
            st.markdown('<div style="padding:60px 12px;text-align:center;border:1px dashed #2a4967;border-radius:14px;background:#0a1727"><div style="font-size:3rem">◉</div><div style="font-size:1.2rem;font-weight:800;color:#e9f2ff">Awaiting image analysis</div><div style="color:#91a8c3;margin-top:7px">Upload an image and run detection.</div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

elif page=="📊 Analytics":
    st.markdown('<div class="section-title">Detection Analytics</div><div class="muted">Summary for the most recently processed image.</div><br>',unsafe_allow_html=True)
    if "dv_detections" not in st.session_state: st.warning("Run a detection first in the Detection Lab.")
    else:
        df=pd.DataFrame(st.session_state["dv_detections"])
        c1,c2,c3=st.columns(3)
        for col,label,value in [(c1,"Total detections",len(df)),(c2,"Unique classes",df["Object"].nunique() if len(df) else 0),(c3,"Average confidence",f'{df["Confidence (%)"].mean():.1f}%' if len(df) else "—")]:
            with col: st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',unsafe_allow_html=True)
        if len(df):
            a,b=st.columns(2)
            with a:
                st.markdown('<div class="panel">**Object counts by class**',unsafe_allow_html=True); st.bar_chart(df["Object"].value_counts()); st.markdown('</div>',unsafe_allow_html=True)
            with b:
                st.markdown('<div class="panel">**Confidence by detection**',unsafe_allow_html=True); st.bar_chart(df["Confidence (%)"]); st.markdown('</div>',unsafe_allow_html=True)
            st.dataframe(df,use_container_width=True,hide_index=True)

else:
    st.markdown('<div class="section-title">Project Details</div><div class="muted">Overview, workflow and viva preparation.</div><br>',unsafe_allow_html=True)
    a,b=st.columns(2,gap="large")
    with a:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown("### 🎯 Objective\nDetect common object classes in uploaded images and visualize their locations with bounding boxes.")
        st.markdown("### 🧰 Technology stack\n- Python\n- Streamlit\n- Ultralytics YOLOv8 nano\n- Pillow\n- Pandas")
        st.markdown("### 🔄 Workflow\n1. Upload image.\n2. Run pretrained model inference.\n3. Extract labels, confidence scores and box coordinates.\n4. Draw boxes and show analytics.\n5. Download annotated image.")
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.markdown("### 🧠 What is YOLO?\nYOLO means **You Only Look Once**. It is a family of object-detection models that predicts object classes and bounding boxes in a single model pass.")
        st.markdown("### ⚠️ Limitations\nDetection depends on image quality, object size and training data. False detections and missed objects are possible. This prototype is not for operational defence or safety-critical use.")
        st.markdown("### 🎓 Viva quick answers\n**Output:** class, confidence and bounding box.  \n**Confidence:** model score, not a guarantee.  \n**Streamlit:** rapid Python web-app framework.  \n**Future scope:** evaluate on labelled data and compare models.")
        st.markdown('</div>',unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#7189a7;font-size:12px">DEFENCEVISION AI · OBJECT DETECTION MINI PROJECT · EDUCATIONAL USE</div>',unsafe_allow_html=True)
