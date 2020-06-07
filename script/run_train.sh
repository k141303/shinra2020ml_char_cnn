LANG=('ar' 'bg' 'ca' 'cs' 'da' 'de' 'el' 'en' 'es' 'fa' \
      'fi' 'fr' 'he' 'hi' 'hu' 'id' 'it' 'ko' 'nl' 'no' \
      'pl' 'pt' 'ro' 'ru' 'sv' 'th' 'tr' 'uk' 'vi' 'zh')

for lang in ${LANG[@]}
do
python3 train.py --batch_size 512 \
                --epoch 15 \
                --lr 5e-3 \
                --adam_eps 1e-8 \
                --output ./result \
                --num_workers 8 \
                --lang $lang
done
