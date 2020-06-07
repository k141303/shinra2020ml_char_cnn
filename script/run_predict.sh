LANG=('ar' 'bg' 'ca' 'cs' 'da' 'de' 'el' 'en' 'es' 'fa' \
      'fi' 'fr' 'he' 'hi' 'hu' 'id' 'it' 'ko' 'nl' 'no' \
      'pl' 'pt' 'ro' 'ru' 'sv' 'th' 'tr' 'uk' 'vi' 'zh')

for lang in ${LANG[@]}
do
python3 predict.py -i $1 \
                   -o ./prediction \
                   --lang $lang \
                   --pretrained_model ./result
done
