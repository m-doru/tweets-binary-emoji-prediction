file=$1

sort  $file | uniq -u > ${file}.sorted

file=${file}.sorted


#create one token from : - * as it is a kiss
sed -i -e "s/: - */:-*/g" $file


# create one token from multiple closed brackets
pattern="( ( ( ( ( ( ( ( ("
repl_pat="((((((((("

for i in {1..6}; do
    pattern=${pattern:2}
    repl_pat=${repl_pat:1}
    echo $repl_pat

    sed -i -e "s/${pattern}/${repl_pat}/g" $file
done

pattern=") ) ) ) ) ) ) ) ) ) ) ) ) )"
repl_pat="))))))))))))))"


for i in {1..11}; do
    pattern=${pattern:2}
    repl_pat=${repl_pat:1}
    echo $repl_pat

    sed -i -e "s/${pattern}/${repl_pat}/g" $file
done
