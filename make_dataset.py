import os
import glob
import json
import csv

# Función para extraer un valor del technicalSheet dado el campo
def get_technical_value(tech_sheet, field_name):
    for item in tech_sheet:
        if item.get("field") == field_name:
            return item.get("value")
    return None

def main():
    # Directorio donde se encuentran los archivos JSON
    json_dir = "./json_data"  # Cambia esto según tu ruta de archivos
    output_csv = "dataset.csv"

    # Lista de archivos JSON en el directorio
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    # Encabezados del CSV
    headers = [
        "ID",
        "Tipo de inmueble",
        "Precio (admin_included)",
        "Estrato",
        "Baños",
        "Habitaciones",
        "Parqueaderos",
        "Area",
        "Area Privada",
        "Piso",
        "Facilities",
        "Descripción",
        "Antiguedad",
        "Estado"
    ]
    
    rows = []
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Navegamos la estructura: props -> pageProps -> data
            prop_data = data.get("props", {}).get("pageProps", {}).get("data", {})
            tech_sheet = prop_data.get("technicalSheet", [])
            
            # Extraer los campos
            id = prop_data.get("id")
            tipo_inmueble = get_technical_value(tech_sheet, "property_type_name")
            precio_admin = prop_data.get("price", {}).get("admin_included")
            estrato = get_technical_value(tech_sheet, "stratum")
            banos = get_technical_value(tech_sheet, "bathrooms")
            habitaciones = get_technical_value(tech_sheet, "bedrooms")
            parqueaderos = get_technical_value(tech_sheet, "garage")
            area = get_technical_value(tech_sheet, "m2Built")
            area_privada = get_technical_value(tech_sheet, "m2Terrain")
            antiguedad = get_technical_value(tech_sheet, "constructionYear")
            estado = get_technical_value(tech_sheet, "construction_state_name")

            piso = prop_data.get("floor")
            
            # Extraer facilities: se unen los nombres de las facilities
            facilities_list = prop_data.get("facilities", [])
            facilities = ", ".join([fac.get("name") for fac in facilities_list])
            
            descripcion = prop_data.get("description", "").replace("\n", " ").replace("\r", " ")
            
            
            # Agregar la fila
            rows.append([
                id,
                tipo_inmueble,
                precio_admin,
                estrato,
                banos,
                habitaciones,
                parqueaderos,
                area,
                area_privada,
                piso,
                facilities,
                descripcion,
                antiguedad,
                estado
            ])
            
        except Exception as e:
            print(f"Error procesando el archivo {file_path}: {e}")

    # Escribir el CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f"CSV generado exitosamente: {output_csv}")

if __name__ == "__main__":
    main()