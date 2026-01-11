from tools.set_security_code import set_security_code
from security_code_verification import security_code_verification
from tools.set_routine import set_routine
from ExerciseDetectionSystem import detect_exercise


def main():
    # COnfigurar el código de securidad
    set_security_code()

    # Verificar código de seguridad
    resultado = security_code_verification(csv_path="data/security_code_color_ranges.csv" , camera_index=0, width=1280, height=720)
    
    # Si se ha verificado correctamente, se accede al programa de ejercicio
    if resultado is True:

        # COnfigurar rutina personalizada
        set_routine()

        # Programa principal
        detect_exercise()

    elif resultado is False:
        print("Usuario canceló")

    else:  # None
        print("Error o interrupción")


if __name__ == "__main__":
    main()