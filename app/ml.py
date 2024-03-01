import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

skin_conditions = {
    "Eczema": {"over_the_counter": ["Hydrocortisone cream", "Calamine lotion", "Emollients"], "prescription": ["Topical calcineurin inhibitors", "Oral antihistamines", "Systemic corticosteroids"]},
    "Melanoma": {"over_the_counter": [], "prescription": ["Immunotherapies such as pembrolizumab, ipilimumab, and nivolumab", "Chemo therapies such as dacarbazine, cisplatin, carboplatin, taxanes, and alkylators"]},
    "Psoriasis": {"over_the_counter": ["Coal tar preparations", "Salicylic acid", "Moisturizers"], "prescription": ["Biological drugs such as adalimumab, etanercept, infliximab, ustekinumab, secukinumab, brodalumab, and apremilast", "Retinoids like acitretin and tazarotene.", "Methotrexate"]},
    "Basal Cell Carcinoma": {"over_the_counter": ["Imiquimod cream", "Fluorouracil topical solution"], "prescription": ["Surgeries such as Mohs micrographic surgery, excisions, and grafting techniques", "Radiotherapy"]},
    "Seborrheic Keratoses": {"over_the_counter": ["Ketoconazole shampoo", "Salicylic acid solutions", "Coal tar preparations"], "prescription": ["cryotherapy (freezing)", "curettage (scraping)", "laser therapy for removal."]}
}


BATCH_SIZE = 32
IMAGE_SIZE = 256

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    # 'E:\\7th Semester\\Final Year Project\New folder\\Skin-Disease-Detection-System-Using-CNN-Model-main\\skin_dataset',
    
    'E:\\7thSem_Project\\SkinDiseasePrediction\\skin_dataset',
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    prescription = ' ,'.join([str(medicine) for medicine in skin_conditions[predicted_class]["prescription"]])
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, prescription, confidence

def get_treatments(predicted_condition):
    treatments = skin_conditions[predicted_condition]
    medicines = treatments["over_the_counter"] + treatments["prescription"]
    print("Medicines for {}:".format(predicted_condition))
    for medicine in medicines:
        print("- {}", medicine)


def main():
    model = tf.keras.models.load_model('E:\7thSem_Project\SkinDiseasePrediction\trained.h2\\')

    img_file_path = 'path_to_input_image.jpg'
    predicted_class, _ = predict(model, img_file_path)
    get_treatments(predicted_class)

if __name__ == "__main__":
    main()
