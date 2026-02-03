import streamlit as st
import torch
import numpy as np
import shap
from scipy.special import softmax
from functools import lru_cache
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit.components.v1 as components


# ==========================================
# 1. SETUP & CACHING
# ==========================================

st.set_page_config(page_title="Analysis", layout="wide")


@st.cache_resource
def load_resources():
    try:
        tokenizer = AutoTokenizer.from_pretrained("./results")
        model = AutoModelForSequenceClassification.from_pretrained(
            "./results",
            num_labels=2,
            problem_type="single_label_classification",
            local_files_only=True
        )
        return model, tokenizer

    except Exception as e:
        st.error(f"Error loading model from './results'. Details: {e}")
        return None, None


model, tokenizer = load_resources()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. SHAP WRAPPER
# ==========================================

class DocumentSHAPWrapper:

    def __init__(
        self,
        model,
        tokenizer,
        device,
        class_names=("Human", "AI"),
        max_length=512,
        overlap=30
    ):

        self.model = model.to(device)
        self.model.eval()
        torch.set_grad_enabled(False)

        self.tokenizer = tokenizer
        self.device = device
        self.class_names = class_names

        self.max_length = min(max_length, tokenizer.model_max_length)
        self.overlap = overlap
        self.chunk_size = self.max_length - 2

        self.explainer = shap.Explainer(
            self._predict_document,
            masker=shap.maskers.Text(self.tokenizer),
            algorithm="partition",
            output_names=self.class_names,
        )


    def _split_into_chunks(self, text):

        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=50000
        )

        stride = max(1, self.chunk_size - self.overlap)

        chunks = []

        for i in range(0, len(tokens), stride):

            part = tokens[i:i + self.chunk_size]

            if len(part) < 3:
                continue

            chunks.append(
                self.tokenizer.decode(
                    part,
                    skip_special_tokens=True
                )
            )

        return chunks if chunks else [text]


    def _split_into_chunks_no_overlap(self, text):

        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=50000
        )

        stride = self.chunk_size

        chunks = []

        for i in range(0, len(tokens), stride):

            part = tokens[i:i + self.chunk_size]

            if len(part) < 3:
                continue

            chunks.append(
                self.tokenizer.decode(
                    part,
                    skip_special_tokens=True
                )
            )

        return chunks if chunks else [text]


    @lru_cache(maxsize=256)
    def _cached_chunk_encodings(self, text):

        text = " ".join(text.split())

        chunks = self._split_into_chunks(text)

        return self.tokenizer(
            chunks,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )


    def _predict_document(self, texts):

        all_avg_logits = []

        with torch.inference_mode():

            for text in texts:

                if not text.strip():

                    all_avg_logits.append(
                        np.zeros(len(self.class_names)) + 1e-6
                    )

                    continue

                encodings = self._cached_chunk_encodings(text)

                encodings = {
                    k: v.to(self.device)
                    for k, v in encodings.items()
                }

                outputs = self.model(**encodings)

                logits = outputs.logits

                avg_logits = torch.mean(
                    logits,
                    dim=0
                ).detach().cpu().numpy()

                all_avg_logits.append(avg_logits)

        return np.array(all_avg_logits)


    # ==================================================
    # ✅ IMPROVED VISUALIZATION WITH WHITE BACKGROUND
    # ==================================================

    def _generate_token_html(self, tokens, values):
        """
        Generate clean SHAP visualization with proper red/blue coloring
        Red = pushes toward AI, Blue = pushes toward Human
        """

        max_val = np.max(np.abs(values))

        if max_val == 0:
            max_val = 1e-9


        def shap_color(val):
            """
            Positive values (toward AI) = Red
            Negative values (toward Human) = Blue
            """
            norm = val / max_val
            norm = np.clip(norm, -1, 1)

            if norm > 0:
                # Red for AI (positive values)
                intensity = norm
                r = 255
                g = int(240 * (1 - intensity * 0.8))
                b = int(240 * (1 - intensity * 0.8))
            else:
                # Blue for Human (negative values)
                intensity = -norm
                r = int(240 * (1 - intensity * 0.8))
                g = int(240 * (1 - intensity * 0.8))
                b = 255

            return f"rgb({r},{g},{b})"


        html_parts = []

        for token, val in zip(tokens, values):

            color = shap_color(val)

            token_esc = (
                token
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace(" ", "&nbsp;")
            )

            html_parts.append(f"""<span
                title="SHAP value: {val:.5f}"
                style="
                    background-color:{color};
                    color:#1a1a1a;
                    padding:2px 1px;
                    margin:0;
                    font-weight:500;
                    display:inline;
                    transition:all 0.2s ease;
                    line-height:1.8;
                "
                onmouseover="this.style.transform='scale(1.05)'"
                onmouseout="this.style.transform='scale(1)'"
            >{token_esc}</span>""")

        return "".join(html_parts)


    def _wrap_html(self, content):
        """
        Clean white background container
        """

        html = f"""
        <div style="
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', sans-serif;
            padding: 24px;
            border-radius: 8px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        ">

            <div style="
                text-align: center;
                padding: 12px;
                margin-bottom: 20px;
                border-radius: 6px;
                background: #f9fafb;
                border: 1px solid #e5e7eb;
                font-size: 13px;
            ">

                <span style="
                    display: inline-block;
                    padding: 4px 12px;
                    background: #dbeafe;
                    color: #1e40af;
                    border-radius: 4px;
                    font-weight: 600;
                    margin-right: 8px;
                ">Human</span>
                
                <span style="
                    display: inline-block;
                    padding: 4px 12px;
                    background: #fee2e2;
                    color: #991b1b;
                    border-radius: 4px;
                    font-weight: 600;
                    margin-left: 8px;
                ">AI</span>

                <div style="
                    margin-top: 8px;
                    font-size: 12px;
                    color: #6b7280;
                ">
                    Darker colors indicate stronger contributions
                </div>

            </div>

            <div style="
                font-size: 15px;
                line-height: 2;
                word-break: break-word;
                padding: 16px;
                background: #fafafa;
                border-radius: 6px;
                border: 1px solid #e5e7eb;
            ">

                {content}

            </div>

        </div>
        """

        return html


    # ==================================================
    # REMAINING CODE (UNCHANGED)
    # ==================================================

    def generate_chunked_html(self, text, max_evals=200):

        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

        if len(tokens) <= self.chunk_size:

            explanation = self.explainer([text], max_evals=max_evals)

            tokens = explanation.data[0]
            values = explanation.values[0, :, 1]

            html_content = self._generate_token_html(tokens, values)

            return self._wrap_html(html_content), 1


        chunks = self._split_into_chunks_no_overlap(text)

        all_token_html = []


        for chunk in chunks:

            explanation = self.explainer([chunk], max_evals=max_evals)

            tokens = explanation.data[0]
            values = explanation.values[0, :, 1]

            chunk_html = self._generate_token_html(tokens, values)

            all_token_html.append(chunk_html)


        combined_html = " ".join(all_token_html)

        return self._wrap_html(combined_html), len(chunks)


    # ==================================================
    # SHAP-BASED PREDICTION RECONSTRUCTION
    # ==================================================

    def _compute_shap_prediction(self, explanation, class_id):
        """
        Reconstruct prediction from SHAP values
        """
        base_value = explanation.base_values[0, class_id]
        shap_sum = explanation.values[0, :, class_id].sum()
        reconstructed_logit = base_value + shap_sum
        
        all_logits = []
        for i in range(len(self.class_names)):
            base = explanation.base_values[0, i]
            shap_s = explanation.values[0, :, i].sum()
            all_logits.append(base + shap_s)
        
        reconstructed_probs = softmax(all_logits)
        return reconstructed_logit, reconstructed_probs


    def _compute_certainty_per_class(self, explanation):
        """
        Compute NORMALIZED certainty for each class.
        Certainties sum to 1.0 (100%).
        """
        raw_certainties = {}
        
        for class_id, class_name in enumerate(self.class_names):
            shap_values = explanation.values[0, :, class_id]
            total_contribution = np.abs(shap_values).sum()
            
            positive_vals = shap_values[shap_values > 0]
            negative_vals = shap_values[shap_values < 0]
            
            pos_sum = positive_vals.sum() if len(positive_vals) > 0 else 0
            neg_sum = abs(negative_vals.sum()) if len(negative_vals) > 0 else 0
            
            if pos_sum + neg_sum > 0:
                consistency = pos_sum / (pos_sum + neg_sum)
            else:
                consistency = 0.5
            
            _, reconstructed_probs = self._compute_shap_prediction(explanation, class_id)
            class_prob = reconstructed_probs[class_id]
            
            threshold = np.percentile(np.abs(shap_values), 75)
            num_strong = np.sum(np.abs(shap_values) > threshold)
            total_feats = len(shap_values)
            strong_ratio = num_strong / total_feats if total_feats > 0 else 0
            
            # Weighted raw certainty
            raw_certainty = (
                0.50 * class_prob +
                0.30 * consistency +
                0.10 * min(total_contribution / 10, 1.0) +
                0.10 * strong_ratio
            )
            raw_certainties[class_name] = raw_certainty
        
        # Normalize so certainties sum to 1.0
        total = sum(raw_certainties.values())
        if total > 0:
            return {k: v / total for k, v in raw_certainties.items()}
        return {k: 1.0 / len(self.class_names) for k in self.class_names}


    def explain(self, text, max_evals=200):

        if not text.strip():
            return None, None, None, None, 0, None

        explanation = self.explainer([text], max_evals=max_evals)

        # Compute certainty scores
        certainty_per_class = self._compute_certainty_per_class(explanation)

        html_viz, num_chunks = self.generate_chunked_html(
            text,
            max_evals
        )

        return None, None, html_viz, num_chunks, certainty_per_class


# ==========================================
# 3. UI
# ==========================================

st.title("Text Analysis")

st.markdown(
    "Enter text below. The system will provide prediction and SHAP explanation."
)

text_input = st.text_area("Input Text", value=" ", height=200)


if st.button("Analyze", type="primary"):

    if not model:
        st.error("Model failed to load.")

    elif not text_input.strip():
        st.warning("Please enter some text.")

    else:

        wrapper = DocumentSHAPWrapper(
            model,
            tokenizer,
            device
        )


        with st.spinner("Predicting..."):

            logits = wrapper._predict_document([text_input])[0]

            probs = softmax(logits)

            pred_id = np.argmax(probs)

            pred = wrapper.class_names[pred_id]

            # Clear GPU cache after prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Show quick prediction result
            st.subheader("Quick Prediction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Class", pred)
            
            with col2:
                st.metric("Probability", f"{probs[pred_id]:.2%}")


        with st.spinner("Running SHAP analysis..."):

            _, _, html_viz, num_chunks, certainty_per_class = wrapper.explain(
                text_input,
                max_evals=250
            )

            # Release GPU memory after completion
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Show detailed analysis results
            st.subheader("Detailed Analysis")
            
            # Create table for probabilities and certainties
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Class**")
                for name in wrapper.class_names:
                    st.write(name)
            
            with col2:
                st.markdown("**Probability**")
                for i, name in enumerate(wrapper.class_names):
                    st.write(f"{probs[i]:.2%}")
            
            with col3:
                st.markdown("**Certainty**")
                for name in wrapper.class_names:
                    st.write(f"{certainty_per_class[name]:.2%}")


            st.subheader("Visualization")


            token_count = len(
                tokenizer.encode(
                    text_input,
                    add_special_tokens=False
                )
            )

            st.info(
                f"Tokens: {token_count} | Chunks: {num_chunks}"
            )


            estimated_lines = max(
                5,
                len(text_input) // 80 + 1
            )

            viz_height = min(
                200 + estimated_lines * 50,
                2000
            )


            components.html(
                html_viz,
                height=viz_height,
                scrolling=False
            )