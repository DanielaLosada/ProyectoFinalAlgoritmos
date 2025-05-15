import subprocess

def ejecutar_requerimiento(requerimiento):
    try:
        subprocess.run(["python", f"Requerimiento{requerimiento}/main.py"], check=True)
    except subprocess.CalledProcessError:
        print(f"Error al ejecutar requerimiento {requerimiento}.")


def menu():
    while True:
        print("\n--- MENÚ DE REQUERIMIENTOS ---")
        print("1. Ejecutar requerimiento 1")
        print("2. Ejecutar requerimiento 2")
        print("3. Ejecutar requerimiento 3")
        print("4. Ejecutar requerimiento 5")
        print("5. Ejecutar todos los requerimientos (1, 2, 3, 5)")
        print("6. Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            ejecutar_requerimiento(1)
        elif opcion == "2":
            ejecutar_requerimiento(2)
        elif opcion == "3":
            ejecutar_requerimiento(3)
        elif opcion == "4":
            ejecutar_requerimiento(5)
        elif opcion == "5":
            for req in [1, 2, 3, 5]:
                ejecutar_requerimiento(req)
        elif opcion == "6":
            print("Saliendo...")
            break
        else:
            print("Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    menu()
