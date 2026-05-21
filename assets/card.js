const flashcards = [
    {
        question: "SARIMA-HOLTWINTERS-LSTM",
        hint: "Modelo#1",
        answer: "Modelo que combinado mejora el pronóstico de series temporales. Integra patrones estacionales, tendencias y aprendizaje profundo, logrando predicciones más precisas en datos complejos y no lineales."
    },
    {
        question: "GRU-TCN-BILstm",
        hint: "Modelo#2",
        answer: "Modelo de aprendizaje profundo, para analizar series temporales complejas. Captura dependencias a corto y largo plazo, patrones secuenciales y relaciones no lineales, mejorando la precisión en predicciones y detección de tendencias."
    },
    {
        question: "Random-Forest",
        hint: "Modelo#3",
        answer: "Modelo de aprendizaje automático basado en múltiples árboles de decisión que trabaja mediante ensambles mejorando la precisión, reduce el sobreajuste. Eficaz para clasificación y predicción, manejando grandes volúmenes de datos y relaciones complejas entre variables."
    },
    {
        question: "VAR-GARCH",
        hint: "Modelo#4",
        answer: "Modelo estadístico para analizar relaciones dinámicas entre múltiples variables y modelar la volatilidad en series temporales. Útil para pronósticos financieros y detección de cambios en el comportamiento de los datos. "
    }, 
        {
        question: "PASS-THROUGH",
        hint: "Modelo#5",
        answer: "Modelo económico utilizado para medir cómo las variaciones de tipo de cambio se trasladan al precio final de bienes y servicios. Permite analizar el impacto de factores externos sobre la inflación o tipo de cambio."
    },
        {
        question: "CINEMÁTICA/TERMODINÁMICA",
        hint: "Modelo#6",
        answer: "movimiento de cuerpos y las leyes que lo describen, junto con los procesos de transferencia y transformación de energía y calor. Fundamentales para analizar sistemas físicos/industriales, etc. "
    }
];

function initFlashcards(cards) {
    let currentIndex = 0;
    let isAnimating = false;
    const container = document.getElementById("card-container");
    const nextBtn = document.getElementById("next-btn");
    const prevBtn = document.getElementById("prev-btn");

    const createCard = ({ question, hint, answer }) => {
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `
            <h2>${question}</h2>
            <p>${hint}</p>
            <p class="answer">${answer}</p>
        `;
        card.querySelector(".answer").addEventListener("click", (e) => {
            e.target.classList.add("revealed");
        });
        return card;
    };

    const showCard = (index, direction) => {
        if (isAnimating) return;
        isAnimating = true;
        const newCard = createCard(cards[index]);
        if (direction === "prev") newCard.classList.add("enter-left");
        container.appendChild(newCard);

        requestAnimationFrame(() => {
            newCard.classList.add("show");
            if (direction === "prev") newCard.classList.remove("enter-left");
        });

        const oldCard = container.children.length > 1 ? container.firstChild : null;
        if (oldCard) {
            oldCard.classList.remove("show");
            oldCard.classList.add(direction === "next" ? "exit-left" : "exit-right");
            setTimeout(() => { oldCard.remove(); isAnimating = false; }, 1000);
        } else {
            setTimeout(() => { isAnimating = false; }, 1000);
        }
    };

    nextBtn.addEventListener("click", () => {
        currentIndex = (currentIndex + 1) % cards.length;
        showCard(currentIndex, "next");
    });

    prevBtn.addEventListener("click", () => {
        currentIndex = (currentIndex - 1 + cards.length) % cards.length;
        showCard(currentIndex, "prev");
    });

    showCard(currentIndex, "next");
}

initFlashcards(flashcards);