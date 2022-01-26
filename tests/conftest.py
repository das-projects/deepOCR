import json
import shutil
import tempfile
from io import BytesIO

import fitz
import hdf5storage
import numpy as np
import pytest
import requests
import scipy.io as sio


@pytest.fixture(scope="session")
def mock_vocab():
    return ('3K}7eé;5àÎYho]QwV6qU~W"XnbBvcADfËmy.9ÔpÛ*{CôïE%M4#ÈR:g@T$x?0î£|za1ù8,OG€P-kçHëÀÂ2É/ûIJ\'j'
            '(LNÙFut[)èZs+&°Sd=Ï!<â_Ç>rêi`l')


@pytest.fixture(scope="session")
def mock_pdf(tmpdir_factory):

    doc = fitz.open()

    page = doc.new_page()
    page.insert_text(fitz.Point(50, 100), "I am a jedi!", fontsize=20)
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 100), "No, I am your father.", fontsize=20)

    # Save the PDF
    fn = tmpdir_factory.mktemp("data").join("mock_pdf_file.pdf")
    with open(fn, 'wb') as f:
        doc.save(f)

    return str(fn)


@pytest.fixture(scope="session")
def mock_text_box_stream():
    url = 'https://www.pngitem.com/pimgs/m/357-3579845_love-neon-loveislove-word-text-typography-freetoedit-picsart.png'
    return requests.get(url).content


@pytest.fixture(scope="session")
def mock_text_box(mock_text_box_stream, tmpdir_factory):
    file = BytesIO(mock_text_box_stream)
    fn = tmpdir_factory.mktemp("data").join("mock_text_box_file.png")
    with open(fn, 'wb') as f:
        f.write(file.getbuffer())
    return str(fn)


@pytest.fixture(scope="session")
def mock_image_stream():
    url = "https://miro.medium.com/max/3349/1*mk1-6aYaf_Bes1E3Imhc0A.jpeg"
    return requests.get(url).content


@pytest.fixture(scope="session")
def mock_image_path(mock_image_stream, tmpdir_factory):
    file = BytesIO(mock_image_stream)
    folder = tmpdir_factory.mktemp("images")
    fn = folder.join("mock_image_file.jpeg")
    with open(fn, 'wb') as f:
        f.write(file.getbuffer())
    return str(fn)


@pytest.fixture(scope="session")
def mock_image_folder(mock_image_stream, tmpdir_factory):
    file = BytesIO(mock_image_stream)
    folder = tmpdir_factory.mktemp("images")
    for i in range(5):
        fn = folder.join("mock_image_file_" + str(i) + ".jpeg")
        with open(fn, 'wb') as f:
            f.write(file.getbuffer())
    return str(folder)


@pytest.fixture(scope="session")
def mock_detection_label(tmpdir_factory):
    folder = tmpdir_factory.mktemp("labels")
    labels = {}
    for idx in range(5):
        labels[f"mock_image_file_{idx}.jpeg"] = {
            "img_dimensions": (800, 600),
            "img_hash": "dummy_hash",
            "polygons": [
                [[1, 2], [1, 3], [2, 1], [2, 3]],
                [[10, 20], [10, 30], [20, 10], [20, 30]],
                [[3, 2], [3, 3], [4, 1], [4, 3]],
                [[30, 20], [30, 30], [40, 10], [40, 30]],
            ],
        }

    labels_path = folder.join('labels.json')
    with open(labels_path, 'w') as f:
        json.dump(labels, f)
    return str(labels_path)


@pytest.fixture(scope="session")
def mock_recognition_label(tmpdir_factory):
    label_file = tmpdir_factory.mktemp("labels").join("labels.json")
    label = {
        "mock_image_file_0.jpeg": "I",
        "mock_image_file_1.jpeg": "am",
        "mock_image_file_2.jpeg": "a",
        "mock_image_file_3.jpeg": "jedi",
        "mock_image_file_4.jpeg": "!",
    }
    with open(label_file, 'w') as f:
        json.dump(label, f)
    return str(label_file)


@pytest.fixture(scope="session")
def mock_ocrdataset(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp("dataset")
    label_file = root.join("labels.json")
    label = {
        "mock_image_file_0.jpg": {
            "typed_words": [
                {'value': 'I', 'geometry': (.2, .2, .1, .1, 0)},
                {'value': 'am', 'geometry': (.5, .5, .1, .1, 0)},
            ]
        },
        "mock_image_file_1.jpg": {
            "typed_words": [
                {'value': 'a', 'geometry': (.2, .2, .1, .1, 0)},
                {'value': 'jedi', 'geometry': (.5, .5, .1, .1, 0)},
            ]
        },
        "mock_image_file_2.jpg": {
            "typed_words": [
                {'value': '!', 'geometry': (.2, .2, .1, .1, 0)},
            ]
        }
    }
    with open(label_file, 'w') as f:
        json.dump(label, f)

    file = BytesIO(mock_image_stream)
    image_folder = tmpdir_factory.mktemp("images")
    for i in range(3):
        fn = image_folder.join(f"mock_image_file_{i}.jpg")
        with open(fn, 'wb') as f:
            f.write(file.getbuffer())

    return str(image_folder), str(label_file)


@pytest.fixture(scope="session")
def mock_ic13(tmpdir_factory, mock_image_stream):
    file = BytesIO(mock_image_stream)
    image_folder = tmpdir_factory.mktemp("images")
    label_folder = tmpdir_factory.mktemp("labels")
    labels = ["1309, 2240, 1440, 2341, 'I'\n",
              "800, 2240, 1440, 2341, 'am'\n",
              "500, 2240, 1440, 2341, 'a'\n",
              "900, 2240, 1440, 2341, 'jedi'\n",
              "400, 2240, 1440, 2341, '!'"]
    for i in range(5):
        fn_l = label_folder.join(f"gt_mock_image_file_{i}.txt")
        with open(fn_l, 'w') as f:
            f.writelines(labels)
        fn_i = image_folder.join(f"mock_image_file_{i}.jpg")
        with open(fn_i, 'wb') as f:
            f.write(file.getbuffer())
    return str(image_folder), str(label_folder)


@pytest.fixture(scope="session")
def mock_imgur5k(tmpdir_factory, mock_image_stream):
    file = BytesIO(mock_image_stream)
    image_folder = tmpdir_factory.mktemp("images")
    label_folder = tmpdir_factory.mktemp("dataset_info")
    labels = {
        "index_id": {
            "YsaVkzl": {
                "image_url": "https://i.imgur.com/YsaVkzl.jpg",
                "image_path": "/path/to/IMGUR5K-Handwriting-Dataset/images/YsaVkzl.jpg",
                "image_hash": "993a7cbb04a7c854d1d841b065948369"
            },
            "wz3wHhN": {
                "image_url": "https://i.imgur.com/wz3wHhN.jpg",
                "image_path": "/path/to/IMGUR5K-Handwriting-Dataset/images/wz3wHhN.jpg",
                "image_hash": "9157426a98ee52f3e1e8d41fa3a99175"
            },
            "BRHSP23": {
                "image_url": "https://i.imgur.com/BRHSP23.jpg",
                "image_path": "/path/to/IMGUR5K-Handwriting-Dataset/images/BRHSP23.jpg",
                "image_hash": "aab01f7ac82ae53845b01674e9e34167"
            }
        },
        "index_to_ann_map": {
            "YsaVkzl": [
                "YsaVkzl_0",
                "YsaVkzl_1",
                "YsaVkzl_2"],
            "wz3wHhN": [
                "wz3wHhN_0",
                "wz3wHhN_1"],
            "BRHSP23": [
                "BRHSP23_0"]
        },
        "ann_id": {
            "YsaVkzl_0": {
                "word": "I",
                "bounding_box": "[605.33, 1150.67, 614.33, 226.33, 81.0]"
            },
            "YsaVkzl_1": {
                "word": "am",
                "bounding_box": "[783.67, 654.67, 521.0, 222.33, 56.67]"
            },
            "YsaVkzl_2": {
                "word": "a",
                "bounding_box": "[959.0, 437.0, 76.67, 201.0, 38.33]"
            },
            "wz3wHhN_0": {
                "word": "jedi",
                "bounding_box": "[783.67, 654.67, 521.0, 222.33, 56.67]"
            },
            "wz3wHhN_1": {
                "word": "!",
                "bounding_box": "[959.0, 437.0, 76.67, 201.0, 38.33]"
            },
            "BRHSP23_0": {
                "word": "jedi",
                "bounding_box": "[783.67, 654.67, 521.0, 222.33, 56.67]"
            }
        }
    }
    label_file = label_folder.join("imgur5k_annotations.json")
    with open(label_file, 'w') as f:
        json.dump(labels, f)
    for index_id in ['YsaVkzl', 'wz3wHhN', 'BRHSP23']:
        fn_i = image_folder.join(f"{index_id}.jpg")
        with open(fn_i, 'wb') as f:
            f.write(file.getbuffer())
    return str(image_folder), str(label_file)


@pytest.fixture(scope="session")
def mock_svhn_dataset(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp('datasets')
    svhn_root = root.mkdir('svhn')
    file = BytesIO(mock_image_stream)
    # ascii image names
    first = np.array([[49], [46], [112], [110], [103]], dtype=np.int16)  # 1.png
    second = np.array([[50], [46], [112], [110], [103]], dtype=np.int16)  # 2.png
    third = np.array([[51], [46], [112], [110], [103]], dtype=np.int16)  # 3.png
    # labels: label is also ascii
    label = {'height': [35, 35, 35, 35], 'label': [1, 1, 3, 7],
             'left': [116, 128, 137, 151], 'top': [27, 29, 29, 26],
             'width': [15, 10, 17, 17]}

    matcontent = {'digitStruct': {'name': [first, second, third], 'bbox': [label, label, label]}}
    # Mock train data
    train_root = svhn_root.mkdir('train')
    hdf5storage.write(matcontent, filename=train_root.join('digitStruct.mat'))
    for i in range(3):
        fn = train_root.join(f'{i+1}.png')
        with open(fn, 'wb') as f:
            f.write(file.getbuffer())
    # Packing data into an archive to simulate the real data set and bypass archive extraction
    archive_path = root.join('svhn_train.tar')
    shutil.make_archive(root.join('svhn_train'), 'tar', str(svhn_root))
    return str(archive_path)


@pytest.fixture(scope="session")
def mock_sroie_dataset(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp('datasets')
    sroie_root = root.mkdir('sroie2019_train_task1')
    annotations_folder = sroie_root.mkdir('annotations')
    image_folder = sroie_root.mkdir("images")
    labels = ["72, 25, 326, 25, 326, 64, 72, 64, 'I'\n",
              "50, 82, 440, 82, 440, 121, 50, 121, 'am'\n",
              "205, 121, 285, 121, 285, 139, 205, 139, 'a'\n",
              "18, 250, 440, 320, 250, 64, 85, 121, 'jedi'\n",
              "400, 112, 252, 84, 112, 84, 75, 88, '!'"]

    file = BytesIO(mock_image_stream)
    for i in range(3):
        fn_i = image_folder.join(f"{i}.jpg")
        with open(fn_i, 'wb') as f:
            f.write(file.getbuffer())
        fn_l = annotations_folder.join(f"{i}.txt")
        with open(fn_l, 'w') as f:
            f.writelines(labels)

    # Packing data into an archive to simulate the real data set and bypass archive extraction
    archive_path = root.join('sroie2019_train_task1.zip')
    shutil.make_archive(root.join('sroie2019_train_task1'), 'zip', str(sroie_root))
    return str(archive_path)


@pytest.fixture(scope="session")
def mock_funsd_dataset(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp('datasets')
    funsd_root = root.mkdir('funsd')
    sub_dataset_root = funsd_root.mkdir('dataset')
    train_root = sub_dataset_root.mkdir('training_data')
    image_folder = train_root.mkdir("images")
    annotations_folder = train_root.mkdir("annotations")
    labels = {
        "form": [{
            "box": [84, 109, 136, 119],
            "text": "I",
            "label": "question",
            "words": [{"box": [84, 109, 136, 119], "text": "I"}],
            "linking": [[0, 37]],
            "id": 0
        },
            {
            "box": [85, 110, 145, 120],
            "text": "am",
            "label": "answer",
            "words": [{"box": [85, 110, 145, 120], "text": "am"}],
            "linking": [[1, 38]],
            "id": 1
        },
            {
            "box": [86, 115, 150, 125],
            "text": "Luke",
            "label": "answer",
            "words": [{"box": [86, 115, 150, 125], "text": "Luke"}],
            "linking": [[2, 44]],
            "id": 2
        }]
    }

    file = BytesIO(mock_image_stream)
    for i in range(3):
        fn_i = image_folder.join(f"{i}.png")
        with open(fn_i, 'wb') as f:
            f.write(file.getbuffer())
        fn_l = annotations_folder.join(f"{i}.json")
        with open(fn_l, 'w') as f:
            json.dump(labels, f)

    # Packing data into an archive to simulate the real data set and bypass archive extraction
    archive_path = root.join('funsd.zip')
    shutil.make_archive(root.join('funsd'), 'zip', str(funsd_root))
    return str(archive_path)


@pytest.fixture(scope="session")
def mock_cord_dataset(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp('datasets')
    cord_root = root.mkdir('cord_train')
    image_folder = cord_root.mkdir("image")
    annotations_folder = cord_root.mkdir("json")
    labels = {
        "dontcare": [],
        "valid_line": [
            {
                "words": [
                    {
                        "quad":
                            {"x2": 270, "y3": 390, "x3": 270, "y4": 390, "x1": 256, "y1": 374, "x4": 256, "y2": 374},
                        "is_key": 0,
                        "row_id": 2179893,
                        "text": "I"
                    }
                ],
                "category": "menu.cnt",
                "group_id": 3
            },
            {
                "words": [
                    {
                        "quad":
                            {"x2": 270, "y3": 418, "x3": 270, "y4": 418, "x1": 258, "y1": 402, "x4": 258, "y2": 402},
                        "is_key": 0,
                        "row_id": 2179894,
                        "text": "am"
                    }
                ],
                "category": "menu.cnt",
                "group_id": 4
            },
            {
                "words": [
                    {
                        "quad":
                            {"x2": 272, "y3": 444, "x3": 272, "y4": 444, "x1": 258, "y1": 428, "x4": 258, "y2": 428},
                        "is_key": 0,
                        "row_id": 2179895,
                        "text": "Luke"
                    }
                ],
                "category": "menu.cnt",
                "group_id": 5
            }]
    }

    file = BytesIO(mock_image_stream)
    for i in range(3):
        fn_i = image_folder.join(f"receipt_{i}.png")
        with open(fn_i, 'wb') as f:
            f.write(file.getbuffer())
        fn_l = annotations_folder.join(f"receipt_{i}.json")
        with open(fn_l, 'w') as f:
            json.dump(labels, f)

    # Packing data into an archive to simulate the real data set and bypass archive extraction
    archive_path = root.join('cord_train.zip')
    shutil.make_archive(root.join('cord_train'), 'zip', str(cord_root))
    return str(archive_path)


@pytest.fixture(scope="session")
def mock_synthtext_dataset(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp('datasets')
    synthtext_root = root.mkdir('SynthText')
    image_folder = synthtext_root.mkdir("8")
    annotation_file = synthtext_root.join('gt.mat')
    labels = {
        "imnames": [[["8/ballet_106_0.jpg"], ["8/ballet_106_1.jpg"], ["8/ballet_106_2.jpg"]]],
        "wordBB": [[np.random.randint(1000, size=(2, 4, 5)) for _ in range(3)]],
        "txt": [np.array([['I      ', 'am\na      ', 'Jedi   ', '!'] for _ in range(3)])],
    }
    # hacky trick to write file into a LocalPath object with scipy.io.savemat
    with tempfile.NamedTemporaryFile(mode='wb', delete=True) as f:
        sio.savemat(f.name, labels)
        shutil.copy(f.name, str(annotation_file))

    file = BytesIO(mock_image_stream)
    for i in range(3):
        fn_i = image_folder.join(f"ballet_106_{i}.jpg")
        with open(fn_i, 'wb') as f:
            f.write(file.getbuffer())

    # Packing data into an archive to simulate the real data set and bypass archive extraction
    archive_path = root.join('SynthText.zip')
    shutil.make_archive(root.join('SynthText'), 'zip', str(synthtext_root))
    return str(archive_path)


@pytest.fixture(scope="session")
def mock_doc_artefacts(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp('datasets')
    doc_root = root.mkdir('artefact_detection')
    labels = {
        '0.jpg':
        [
            {'geometry': [0.94375, 0.4013671875, 0.99375, 0.4365234375],
             'label': 'bar_code'},
            {'geometry': [0.03125, 0.6923828125, 0.07875, 0.7294921875],
             'label': 'qr_code'},
            {'geometry': [0.1975, 0.1748046875, 0.39875, 0.2216796875],
             'label': 'bar_code'}
        ],
        '1.jpg': [
            {'geometry': [0.94375, 0.4013671875, 0.99375, 0.4365234375],
             'label': 'bar_code'},
            {'geometry': [0.03125, 0.6923828125, 0.07875, 0.7294921875],
             'label': 'qr_code'},
            {'geometry': [0.1975, 0.1748046875, 0.39875, 0.2216796875],
             'label': 'background'}
        ],
        '2.jpg': [
            {'geometry': [0.94375, 0.4013671875, 0.99375, 0.4365234375],
             'label': 'logo'},
            {'geometry': [0.03125, 0.6923828125, 0.07875, 0.7294921875],
             'label': 'qr_code'},
            {'geometry': [0.1975, 0.1748046875, 0.39875, 0.2216796875],
             'label': 'photo'}
        ]
    }
    train_root = doc_root.mkdir('train')
    label_file = train_root.join("labels.json")

    with open(label_file, 'w') as f:
        json.dump(labels, f)

    image_folder = train_root.mkdir("images")
    file = BytesIO(mock_image_stream)
    for i in range(3):
        fn = image_folder.join(f"{i}.jpg")
        with open(fn, 'wb') as f:
            f.write(file.getbuffer())
    # Packing data into an archive to simulate the real data set and bypass archive extraction
    archive_path = root.join('artefact_detection.zip')
    shutil.make_archive(root.join('artefact_detection'), 'zip', str(doc_root))
    return str(archive_path)


@pytest.fixture(scope="session")
def mock_iiit5k_dataset(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp('datasets')
    iiit5k_root = root.mkdir('IIIT5K')
    image_folder = iiit5k_root.mkdir('train')
    annotation_file = iiit5k_root.join('trainCharBound.mat')
    labels = {'trainCharBound':
              {"ImgName": ["train/0.png"], "chars": ["I"], "charBB": np.random.randint(50, size=(1, 4))},
              }

    # hacky trick to write file into a LocalPath object with scipy.io.savemat
    with tempfile.NamedTemporaryFile(mode='wb', delete=True) as f:
        sio.savemat(f.name, labels)
        shutil.copy(f.name, str(annotation_file))

    file = BytesIO(mock_image_stream)
    for i in range(1):
        fn_i = image_folder.join(f"{i}.png")
        with open(fn_i, 'wb') as f:
            f.write(file.getbuffer())

    # Packing data into an archive to simulate the real data set and bypass archive extraction
    archive_path = root.join('IIIT5K-Word-V3.tar')
    shutil.make_archive(root.join('IIIT5K-Word-V3'), 'tar', str(iiit5k_root))
    return str(archive_path)


@pytest.fixture(scope="session")
def mock_svt_dataset(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp('datasets')
    svt_root = root.mkdir('svt1')
    labels = """<tagset><image><imageName>img/00_00.jpg</imageName>
    <address>341 Southwest 10th Avenue Portland OR</address><lex>LIVING,ROOM,THEATERS</lex>
    <Resolution x="1280" y="880"/><taggedRectangles><taggedRectangle height="75" width="236" x="375" y="253">
    <tag>LIVING</tag></taggedRectangle></taggedRectangles></image><image><imageName>img/00_01.jpg</imageName>
    <address>1100 Southwest 6th Avenue Portland OR</address><lex>LULA</lex><Resolution x="1650" y="500"/>
    <taggedRectangles><taggedRectangle height="80" width="250" x="450" y="242"><tag>HOUSE</tag></taggedRectangle>
    </taggedRectangles></image><image><imageName>img/00_02.jpg</imageName>
    <address>341 Southwest 10th Avenue Portland OR</address><lex>LIVING,ROOM,THEATERS</lex><Resolution x="850" y="420"/>
    <taggedRectangles><taggedRectangle height="100" width="250" x="350" y="220"><tag>COST</tag></taggedRectangle>
    </taggedRectangles></image></tagset>"""

    with open(svt_root.join("train.xml"), "w") as f:
        f.write(labels)

    image_folder = svt_root.mkdir("img")
    file = BytesIO(mock_image_stream)
    for i in range(3):
        fn = image_folder.join(f"00_0{i}.jpg")
        with open(fn, 'wb') as f:
            f.write(file.getbuffer())
    # Packing data into an archive to simulate the real data set and bypass archive extraction
    archive_path = root.join('svt.zip')
    shutil.make_archive(root.join('svt'), 'zip', str(svt_root))
    return str(archive_path)


@pytest.fixture(scope="session")
def mock_ic03_dataset(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp('datasets')
    ic03_root = root.mkdir('SceneTrialTrain')
    labels = """<tagset><image><imageName>images/0.jpg</imageName><Resolution x="1280" y="880"/><taggedRectangles>
    <taggedRectangle x="174.0" y="392.0" width="274.0" height="195.0" offset="0.0" rotation="0.0"><tag>LIVING</tag>
    </taggedRectangle></taggedRectangles></image><image><imageName>images/1.jpg</imageName>
    <Resolution x="1650" y="500"/>
    <taggedRectangles><taggedRectangle x="244.0" y="440.0" width="300.0" height="220.0" offset="0.0" rotation="0.0">
    <tag>HOUSE</tag></taggedRectangle></taggedRectangles></image><image><imageName>images/2.jpg</imageName>
    <Resolution x="850" y="420"/><taggedRectangles>
    <taggedRectangle x="180.0" y="400.0" width="280.0" height="250.0" offset="0.0" rotation="0.0"><tag>COST</tag>
    </taggedRectangle></taggedRectangles></image></tagset>"""

    with open(ic03_root.join("words.xml"), "w") as f:
        f.write(labels)

    image_folder = ic03_root.mkdir("images")
    file = BytesIO(mock_image_stream)
    for i in range(3):
        fn = image_folder.join(f"{i}.jpg")
        with open(fn, 'wb') as f:
            f.write(file.getbuffer())
    # Packing data into an archive to simulate the real data set and bypass archive extraction
    archive_path = root.join('ic03_train.zip')
    shutil.make_archive(root.join('ic03_train'), 'zip', str(ic03_root))
    return str(archive_path)
