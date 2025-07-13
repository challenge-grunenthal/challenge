import requests


def get_adverse_events(drug_name: str, limit: int = 10):
    url = (
        f"https://api.fda.gov/drug/event.json?"
        f"search=patient.drug.medicinalproduct:{drug_name}&limit={limit}&sort=receivedate:desc"
    )
    response = requests.get(url)
    response = response.json()

    useful_data = []
    for result in response.get("results", []):
        receivedate = result.get("receivedate", "N/A")
        safetyreportid = result.get("safetyreportid", "N/A")
        # Get drug name(s)
        drug_names = []
        if "patient" in result and "drug" in result["patient"]:
            for drug in result["patient"]["drug"]:
                name = drug.get("medicinalproduct", "N/A")
                if "TRAMADOL" in name.upper():
                    drug_names.append(name)
        # Get reactions
        reactions = []
        outcomes = []
        if "patient" in result and "reaction" in result["patient"]:
            for reaction in result["patient"]["reaction"]:
                reactions.append(reaction.get("reactionmeddrapt", "N/A"))
                outcomes.append(reaction.get("reactionoutcome", "N/A"))
        useful_data.append(
            {
                "receivedate": receivedate,
                "safetyreportid": safetyreportid,
                "drug_names": drug_names,
                "reactions": reactions,
                "outcomes": outcomes,
            }
        )
    return useful_data
