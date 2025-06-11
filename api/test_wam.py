import requests
import json

# Example contract
test_contract = {
  "montant": 80000,
  "dureeMois": 11,
  "offresRecues": 1,
  "procedure": "Procédure adaptée",
  "nature": "Marché",
  "formePrix": "Forfaitaire",
  "ccag" : "Pas de CCAG",
  "typeGroupementOperateurs" : "Pas de groupement",
  "sousTraitanceDeclaree": 1,
  "origineFrance": 0,
  "marcheInnovant": 0,
  "idAccordCadre": "2024Z240001F0",
  "tauxAvance": 0,
  "codeCPV_2_3": 45000000
}

# Send request to your localhost API
response = requests.post("http://localhost:8000/api/predict", json=test_contract)

# Print the response
print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=2))
