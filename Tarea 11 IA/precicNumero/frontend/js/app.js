const areaDibujo = document.getElementById('canvas');
const pincel = areaDibujo.getContext('2d');
const botonLimpiar = document.getElementById('clear-btn');
const botonPredecir = document.getElementById('predict-btn');
const salidaResultado = document.getElementById('result');

let dibujando = false;
let redNeuronal = null;

pincel.fillStyle = "white";
pincel.fillRect(0, 0, areaDibujo.width, areaDibujo.height);
pincel.lineWidth = 14;
pincel.lineCap = "round";
pincel.lineJoin = "round";
pincel.strokeStyle = "black";

async function iniciarIA() {
    salidaResultado.innerText = "Cargando IA...";
    try {
        redNeuronal = await tf.loadLayersModel('./modelo_web/model.json');
        salidaResultado.innerText = "-";
        console.log("Modelo cargado correctamente");
    } catch (err) {
        console.error("No se pudo cargar el modelo:", err);
        salidaResultado.innerText = "Error IA";
    }
}

iniciarIA();

areaDibujo.addEventListener('mousedown', (evento) => {
    dibujando = true;
    trazar(evento);
});

areaDibujo.addEventListener('mousemove', trazar);

areaDibujo.addEventListener('mouseup', () => {
    dibujando = false;
    pincel.beginPath();
});

areaDibujo.addEventListener('mouseleave', () => {
    dibujando = false;
    pincel.beginPath();
});

function trazar(evento) {
    if (!dibujando) return;

    const posicion = areaDibujo.getBoundingClientRect();
    const puntoX = evento.clientX - posicion.left;
    const puntoY = evento.clientY - posicion.top;

    pincel.lineTo(puntoX, puntoY);
    pincel.stroke();
    pincel.beginPath();
    pincel.moveTo(puntoX, puntoY);
}

botonLimpiar.addEventListener('click', () => {
    pincel.fillStyle = "white";
    pincel.fillRect(0, 0, areaDibujo.width, areaDibujo.height);
    salidaResultado.innerText = "-";
});

function prepararImagen(lienzoBase) {
    const contextoBase = lienzoBase.getContext('2d');
    const datosImagen = contextoBase.getImageData(0, 0, lienzoBase.width, lienzoBase.height);
    const pixeles = datosImagen.data;

    let limiteIzq = lienzoBase.width;
    let limiteDer = 0;
    let limiteSup = lienzoBase.height;
    let limiteInf = 0;
    let hayTrazo = false;

    for (let fila = 0; fila < lienzoBase.height; fila++) {
        for (let columna = 0; columna < lienzoBase.width; columna++) {
            const posicionPixel = (fila * lienzoBase.width + columna) * 4;

            if (
                pixeles[posicionPixel] < 220 &&
                pixeles[posicionPixel + 1] < 220 &&
                pixeles[posicionPixel + 2] < 220
            ) {
                if (columna < limiteIzq) limiteIzq = columna;
                if (columna > limiteDer) limiteDer = columna;
                if (fila < limiteSup) limiteSup = fila;
                if (fila > limiteInf) limiteInf = fila;
                hayTrazo = true;
            }
        }
    }

    if (!hayTrazo) return null;

    const margen = 40;

    limiteIzq = Math.max(0, limiteIzq - margen);
    limiteDer = Math.min(lienzoBase.width, limiteDer + margen);
    limiteSup = Math.max(0, limiteSup - margen);
    limiteInf = Math.min(lienzoBase.height, limiteInf + margen);

    const anchoRecorte = limiteDer - limiteIzq;
    const altoRecorte = limiteInf - limiteSup;
    const ladoMayor = Math.max(anchoRecorte, altoRecorte);

    const lienzoCuadrado = document.createElement('canvas');
    lienzoCuadrado.width = ladoMayor;
    lienzoCuadrado.height = ladoMayor;

    const contextoCuadrado = lienzoCuadrado.getContext('2d');

    contextoCuadrado.fillStyle = "white";
    contextoCuadrado.fillRect(0, 0, ladoMayor, ladoMayor);

    const espacioX = (ladoMayor - anchoRecorte) / 2;
    const espacioY = (ladoMayor - altoRecorte) / 2;

    contextoCuadrado.drawImage(
        lienzoBase,
        limiteIzq,
        limiteSup,
        anchoRecorte,
        altoRecorte,
        espacioX,
        espacioY,
        anchoRecorte,
        altoRecorte
    );

    const lienzoFinal = document.createElement('canvas');
    lienzoFinal.width = 28;
    lienzoFinal.height = 28;

    const contextoFinal = lienzoFinal.getContext('2d');

    contextoFinal.imageSmoothingEnabled = true;
    contextoFinal.imageSmoothingQuality = 'high';

    contextoFinal.drawImage(
        lienzoCuadrado,
        0,
        0,
        ladoMayor,
        ladoMayor,
        0,
        0,
        28,
        28
    );

    return lienzoFinal;
}

botonPredecir.addEventListener('click', () => {
    if (!redNeuronal) {
        alert("El modelo web aún se está cargando. Espera un segundo.");
        return;
    }

    const imagenLista = prepararImagen(areaDibujo);

    if (!imagenLista) {
        salidaResultado.innerText = "-";
        return;
    }

    tf.tidy(() => {
        let entrada = tf.browser.fromPixels(imagenLista);

        entrada = entrada.mean(2).expandDims(-1);
        entrada = tf.scalar(1.0).sub(entrada.div(tf.scalar(255.0)));

        const filtro = tf.scalar(0.05);
        entrada = tf.where(entrada.greater(filtro), entrada, tf.zerosLike(entrada));

        entrada = entrada.expandDims(0);

        const resultadoIA = redNeuronal.predict(entrada);
        const numeroDetectado = resultadoIA.argMax(1).dataSync()[0];
        const valores = resultadoIA.dataSync();
        const porcentaje = valores[numeroDetectado] * 100;

        console.log(`Predicción: ${numeroDetectado} | Confianza: ${porcentaje.toFixed(2)}%`);

        salidaResultado.innerText = numeroDetectado;
    });
});