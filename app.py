from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from flask import Flask, render_template, request, jsonify,redirect, url_for
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pywhatkit
from datetime import datetime

app = Flask(__name__)

# Dictionnaire pour stocker les navigateurs ouverts
open_drivers = {}
# Charger le tokenizer et le modèle GPT-2 pour le français
model_name = "antoiloui/belgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def selenium_code(site_url):
    s = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=(s))
    driver.maximize_window()
    driver.get(site_url)
    title = driver.title
    return driver, title

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/automation", methods=["POST"])
def run_automation():
    search_key = request.form.get("search_key")
    site_url = f"https://www.{search_key}.com"

    # Ouvrir un nouveau navigateur et le stocker dans le dictionnaire
    driver, title = selenium_code(site_url)
    open_drivers[search_key] = driver

    # Retourner l'objet driver avec le titre dans la réponse JSON
    return jsonify({'bot_message': f"Recherche Google effectuée avec succès. Résultat: {title}", 'driver_key': search_key})
@app.route('/chat', methods=['POST'])
def chat():
    search_query = request.form.get('search_query')
    if search_query:
        if "heure" in search_query.lower():  # Utilisez lower() pour rendre la comparaison insensible à la casse
            # Obtenir l'heure actuelle
            heure_actuelle = datetime.now().strftime("%H:%M")
            # Renvoyer la réponse au format JSON
            return jsonify({'bot_message': f"Il est {heure_actuelle} je vous communique que c est l heure actuele Monsieur ou Madame je vous remercie pour votre attention."})
        else:
            # Générer une réponse avec GPT-2
            inputs = tokenizer.encode("Vous: " + search_query, return_tensors="pt")
            outputs = model.generate(inputs, max_length=150, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.5)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # renvoyer de la reponse du model 
            return jsonify({'bot_message': "je suis à votre disposition pour parler avec vous, je suis une intelligence artificielle " + response + "."})
    else:
        # Gérer le cas où search_query est vide
        return jsonify({'bot_message': "Votre requête est vide. Veuillez entrer quelque chose."})




@app.route('/play/<video>')
def play(video):
    # Faites ce que vous voulez avec le module pywhatkit.playonyt pour lancer la vidéo
    pywhatkit.playonyt(video)
    return jsonify({'bot_message': "En train de jouer " + video + "."})


@app.route("/message_openDriver", methods=["POST"])
def message_openDriver():
    search_key = request.form.get("search_key")
    if search_key in open_drivers:
        # Obtenir le navigateur à partir du dictionnaire
        driver = open_drivers[search_key]
        # Obtenir le titre du navigateur pour un usage démonstratif
        title = driver.title
        return jsonify({'bot_message': f"Le navigateur est ouvert pour la recherche {search_key}. Titre: {title}", 'driver_key': search_key})
    else:
        return jsonify({'bot_message': "Aucun navigateur ouvert pour la recherche spécifiée."})

if __name__ == "__main__":
    app.run(debug=True)
