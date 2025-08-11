from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import read, write, NO_CONVERT
from datetime import datetime
import os

def execute_notebook_with_variables(notebook_path, variables):
    """
    Exécute un notebook Jupyter avec des variables injectées et retourne les résultats.
    :param notebook_path: Chemin relatif du notebook à exécuter (par rapport au dossier notebooks/).
    :param variables: Dictionnaire contenant les variables à injecter dans le notebook.
    :return: Résultats sous forme de dictionnaire.
    """
    try:
        notebook_full_path = os.path.join('./notebooks', notebook_path)

        # Charger le notebook
        with open(notebook_full_path, 'r', encoding='utf-8') as f:
            nb = read(f, NO_CONVERT)

        # Injecter les variables dans les cellules de code
        for cell in nb['cells']:
            if cell.cell_type == 'code':
                for var_name, var_value in variables.items():
                    cell.source = f"{var_name} = {repr(var_value)}\n" + cell.source

        # Exécuter le notebook
        execute_processor = ExecutePreprocessor(timeout=-1, kernel_name='python3')
        execute_processor.preprocess(nb, {})

        # Sauvegarder le notebook exécuté avec un timestamp
        executed_notebook_path = os.path.join('./notebooks', f"executed_{notebook_path.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb")
        with open(executed_notebook_path, 'w', encoding='utf-8') as f:
            write(nb, f)

        print(f"Notebook exécuté et sauvegardé à : {executed_notebook_path}")

        # Lire les résultats depuis une cellule finale (exemple d'un fichier texte généré)
        result_file_path = 'data_file.txt'
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as file:
                data = file.read().split(',')
            os.remove(result_file_path)  # Supprimer le fichier après lecture
        else:
            data = []

        return {"results": data, "status": "success", "notebook": executed_notebook_path}

    except Exception as e:
        print(f"Erreur lors de l'exécution du notebook : {e}")
        return {"error": str(e), "status": "failure"}

