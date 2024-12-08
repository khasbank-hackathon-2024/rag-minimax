import gradio as gr
import os
from dotenv import load_dotenv
from rag_starter_gemini import RAGBot
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize the bot
api_key = os.getenv('GOOGLE_API_KEY')
bot = RAGBot(api_key)

def format_feedback(feedback_dict):
    required_keys = ['timestamp', 'question', 'answer', 'rating', 'comment']
    for key in required_keys:
        if key not in feedback_dict:
            return "Feedback is incomplete."
    return f"""
    🕒 {feedback_dict['timestamp']}
    👤 Асуулт: {feedback_dict['question']}
    🤖 Хариулт: {feedback_dict['answer']}
    ⭐ Үнэлгээ: {feedback_dict['rating']}
    💬 Санал: {feedback_dict['comment']}
    """


class ChatUI:
    def __init__(self):
        self.feedback_history = []
    
    def respond(self, message, history):
        """Get response from RAG bot"""
        return bot.ask(message)
    
    def save_feedback(self, rating, comment, question, answer):
        """Save user feedback"""
        feedback = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer,
            "rating": rating,
            "comment": comment
        }
        self.feedback_history.append(feedback)
        return format_feedback(feedback)

    def create_interface(self):
        """Create Gradio interface"""
        css = """
        .contain { display: flex; flex-direction: column; }
        #component-0 { height: 100%; }
        .gradio-container { height: 800px !important; }
        """
        
        with gr.Blocks(css=css) as demo:
            # Header section
            gr.Markdown(
                """
                # 🏦 Хас Банк Дижитал Туслах
                
                Сайн байна уу? Би Хас Банкны дижитал туслах **"Хас"** байна.
                """,
                elem_classes="custom-markdown"
            )

            # Chat interface
            chatbot = gr.ChatInterface(
                self.respond,
                examples=[
                    "Бэлтгэн нийлүүлэгчийн зээлийн давуу тал юу вэ?",
                    "Үндсэн хөрөнгийн зээлийн хугацаа хэд вэ?",
                    "Монпэй картын жилийн хураамж хэд вэ?",
                    "ЖДБ эрхлэгчдэд ямар зээл олгодог вэ?",
                    "Бизнес эрхлэгчдэд зориулсан сургалт байдаг уу?",
                    "Үндсэн хөрөнгийн зээлийн хүү хэд вэ?",
                    "Хас Банк хэзээ үүссэн бэ?",
                    "Хас Банк хэдэн салбартай вэ?",
                    "Төв салбар",
                    "Захиалгат гүйлгээ гэж юу вэ?",
                    "РэдПойнт оноо хэрхэн цуглуулах вэ?",
                    "Эко хэрэглээний зээлийн нөхцөл ямар байдаг вэ?"
                ],
                title="",
                description="",
                fill_height=True,
                autofocus=True,
                autoscroll=True
            )

            # Feedback section
            with gr.Accordion("💬 Санал хүсэлт", open=False):
                with gr.Row():
                    rating = gr.Slider(
                        minimum=1, 
                        maximum=5, 
                        value=5, 
                        step=1, 
                        label="Үнэлгээ"
                    )
                    comment = gr.Textbox(
                        label="Санал", 
                        placeholder="Таны санал бидэнд чухал...",
                        lines=2
                    )
                
                feedback_btn = gr.Button("Илгээх", variant="primary")
                feedback_output = gr.Markdown()

                def submit_feedback():
                    if len(chatbot.chat_history) > 0:
                        last_interaction = chatbot.chat_history[-1]
                        return self.save_feedback(
                            rating.value,
                            comment.value,
                            last_interaction[0],
                            last_interaction[1]
                        )
                    return "Чат түүх хоосон байна."

                feedback_btn.click(
                    submit_feedback,
                    inputs=[rating, comment],
                    outputs=[feedback_output]
                )

            # Help section
            with gr.Accordion("❓ Тусламж", open=False):
                gr.Markdown(
                    """
                    ## Хэрхэн ашиглах вэ?
                    1. Асуултаа бичээд Enter товч дарна
                    2. Жишээ асуултуудаас сонгож болно
                    3. Чатын түүхийг цэвэрлэх бол 🗑️ товчийг дарна
                    4. Сүүлийн хариултыг засах бол ↩️ товчийг дарна
                    
                    ## ⚠️ Анхаарах зүйлс
                    * Асуултаа тодорхой, ойлгомжтой бичнэ үү
                    * Нэг удаад нэг асуулт асууна уу
                    * Хувийн мэдээлэл оруулахгүй байна уу
                    """
                )

        return demo

def main():
    ui = ChatUI()
    demo = ui.create_interface()
    demo.launch(share=True)

if __name__ == "__main__":
    main() 