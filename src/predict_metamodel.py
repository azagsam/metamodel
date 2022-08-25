import os

from gensim.models import Doc2Vec
from lemmagen3 import Lemmatizer
from tensorflow import keras
from nltk.corpus import stopwords
import string


def filter_text(content, lem_sl, stopwords):
    content_filtered = []
    for token in content.split():
        lemma = lem_sl.lemmatize(token)
        if lemma not in stopwords:
            content_filtered.append(lemma.lower())
    content_filtered = ' '.join(content_filtered)
    content_filtered = ''.join([i for i in content_filtered if not i.isdigit()])  # remove digits
    content_filtered = content_filtered.translate(str.maketrans('', '', string.punctuation))
    return content_filtered


def get_recommended_model(d2v_model, metamodel, text):
    # preprocess and score
    preprocessed_text = filter_text(text, lem_sl, stopwords).split()
    doc_vector = d2v_model.infer_vector(preprocessed_text)
    scores = metamodel.predict([doc_vector])

    # Scores in group of four: verify the correct order: t5-article, graph-based, hybrid-long, sumbasic
    t5_article = scores[:4]
    graph_based = scores[4:8]
    hybrid_long = scores[8:12]
    sumbasic = scores[12:]


if __name__ == '__main__':
    fname = "model/model-large/metamodel"
    d2v_model = Doc2Vec.load(fname)
    lem_sl = Lemmatizer('sl')
    stopwords = set(stopwords.words('slovene'))

    # load metamodel
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    metamodel = keras.models.load_model('model/metamodel-master/model.h5')
    text = "Marsikdo nam zavida hitrost pri oblikovanju koalicije, a volja ljudi je bila jasna. Če ne bi bila, se nam ne bi uspelo tako hitro dogovoriti, kateri so projekti, smeri in vrednote, ki jih bomo skupaj zagovarjali v prihodnji vladi, je ob podpisu pogodbe dejal verjetni mandatar in predsednik Gibanja Svoboda Robert Golob. Nove organizacije vlade, ki jo pogodba predvideva, še ne morejo takoj udejanjiti, saj jih je ustavil SDS-ov predlog za posvetovalni referendum o zakonu o vladi, a Golob zatrjuje, da bodo to storili v prihodnjih mesecih. Na videz se povečuje kompleksnost vlade, ker se dodajajo nova ministrstva, a v resnici so ta nova ministrstva namenjena ravno tistemu, kar bo našo vlado razlikovalo od prejšnjih. Namenjena so ustvarjanju novih priložnosti, projektov in znanj, je pojasnil. Z ministrstvom za visoko šolstvo, znanost in inovacije, ministrstvom za solidarno prihodnost in ministrstvom zeleni preboj bodo po njegovih besedah omogočili, da bo Slovenija kot država odporna proti spremembam, ki jih prinaša prihodnost. Tudi predsednica SD-ja Tanja Fajon je zatrdila, da so oblikovali vlado sprememb. Naš cilj je, da Sloveniji zagotovimo močno gospodarstvo, socialno varnost za vse, skladen regionalni razvoj in Slovenijo v jedru Evrope. Nova vlada bo usmerjena v dvig dodane vrednosti, v zeleni in digitalni prehod ter v močne javne storitve. Tudi v mednarodni politiki želimo vrniti ugled državi, kjer je bil ta poškodovan. Po besedah koordinatorja Levice Luke Mesca je bilo namreč zadnje desetletje desetletje izgubljenih priložnosti, ko je Slovenija prehajala iz krize v krizo. Ta koalicijska pogodba je za dva mandata, da do leta 2030 ljudem organiziramo državo, kakršno si zaslužijo, je dodal."

    sum_model = get_recommended_model(d2v_model, metamodel, text)