"""Generate a sample course PDF with multi-page machine learning lecture notes."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_course_pdf(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    pages_content = [
        # Page 1
        [
            ("CS401: Advanced Machine Learning", 18, 50),
            ("Lecture 1: Supervised Learning Principles", 14, 80),
            ("Supervised learning is an ML paradigm where algorithms learn a mapping from input features X to targets Y using labeled examples.", 10, 120),
            ("In regression tasks, the target variable Y is continuous, such as predicting house prices or temperature.", 10, 140),
            ("In classification tasks, the target variable Y is categorical, such as classifying emails as spam or legitimate.", 10, 160),
            ("Key algorithms include Ordinary Least Squares, Logistic Regression, Decision Trees, and Support Vector Machines.", 10, 180),
            ("Support Vector Machines find an optimal hyperplane that maximizes the margin between different data classes.", 10, 200),
        ],
        # Page 2
        [
            ("CS401: Advanced Machine Learning", 18, 50),
            ("Lecture 2: Artificial Neural Networks", 14, 80),
            ("Artificial neural networks consist of connected layers of artificial neurons inspired by biological brains.", 10, 120),
            ("A perceptron calculates a weighted sum of inputs, adds a bias term, and passes the result through an activation function.", 10, 140),
            ("Non-linear activation functions allow deep networks to approximate complex non-linear functions.", 10, 160),
            ("The Rectified Linear Unit (ReLU) computes f(x) = max(0, x) and helps prevent vanishing gradient issues.", 10, 180),
            ("The Softmax function converts raw network logits into normalized probability distributions across multiple classes.", 10, 200),
        ],
        # Page 3
        [
            ("CS401: Advanced Machine Learning", 18, 50),
            ("Lecture 3: Backpropagation and Training", 14, 80),
            ("Backpropagation is the fundamental algorithm used to calculate gradients of the loss function with respect to network weights.", 10, 120),
            ("It applies the chain rule of calculus backwards from the output layer through hidden layers to the input layer.", 10, 140),
            ("These computed gradients indicate how each individual weight must be adjusted to reduce overall prediction error.", 10, 160),
            ("Automatic differentiation enables efficient implementation of backpropagation in modern deep learning frameworks.", 10, 180),
        ],
        # Page 4
        [
            ("CS401: Advanced Machine Learning", 18, 50),
            ("Lecture 4: Gradient Descent Optimization", 14, 80),
            ("Gradient descent updates model parameters iteratively in the direction opposite to the gradient of the loss function.", 10, 120),
            ("The learning rate hyperparameter determines the magnitude of steps taken towards the minimum of the loss surface.", 10, 140),
            ("Stochastic Gradient Descent (SGD) updates weights using a single random training sample per step, introducing noisy progress.", 10, 160),
            ("Mini-batch Gradient Descent balances efficiency and stability by computing gradients over small subsets of samples (e.g., 32 or 64).", 10, 180),
            ("Adaptive optimizers like Adam combine momentum with per-parameter adaptive learning rates.", 10, 200),
        ],
        # Page 5
        [
            ("CS401: Advanced Machine Learning", 18, 50),
            ("Lecture 5: Overfitting and Regularization", 14, 80),
            ("Overfitting occurs when a machine learning model fits training noise and achieves poor generalization on unseen test data.", 10, 120),
            ("The bias-variance tradeoff balances underfitting (high bias, low variance) against overfitting (low bias, high variance).", 10, 140),
            ("L2 regularization (Ridge) adds a penalty proportional to the square of weight magnitudes to discourage overly large weights.", 10, 160),
            ("L1 regularization (Lasso) penalizes absolute weight magnitudes and encourages sparsity by driving redundant weights to zero.", 10, 180),
            ("Dropout randomly deactivates a fraction of neurons during training passes, preventing complex co-adaptations.", 10, 200),
        ],
        # Page 6
        [
            ("CS401: Advanced Machine Learning", 18, 50),
            ("Lecture 6: Model Evaluation and Validation", 14, 80),
            ("Model performance should be evaluated on an independent validation or test split that was not seen during training.", 10, 120),
            ("Precision measures the proportion of true positive predictions among all positive predictions made by the classifier.", 10, 140),
            ("Recall measures the proportion of actual positive instances that were successfully identified by the classifier.", 10, 160),
            ("The F1 score is the harmonic mean of precision and recall, balancing both metrics when class distributions are imbalanced.", 10, 180),
            ("The Receiver Operating Characteristic (ROC) curve plots true positive rate against false positive rate across decision thresholds.", 10, 200),
        ],
    ]

    for page_idx, elements in enumerate(pages_content, 1):
        # Header (repetitive line across pages)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(72, height - 36, "DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING - COURSE LECTURE NOTES")

        for text, size, y_offset in elements:
            if size >= 14:
                c.setFont("Helvetica-Bold", size)
            else:
                c.setFont("Helvetica", size)
            c.drawString(72, height - y_offset, text)

        # Footer (repetitive line and page number)
        c.setFont("Helvetica", 9)
        c.drawString(72, 36, "CONFIDENTIAL - STUDENT COURSE REFERENCE ONLY")
        c.drawString(width - 120, 36, f"Page {page_idx} of 6")
        c.showPage()

    c.save()
    return output_path


if __name__ == "__main__":
    pdf_path = Path("data/sample_course_notes.pdf")
    create_course_pdf(pdf_path)
    print(f"Sample PDF created at {pdf_path}")

