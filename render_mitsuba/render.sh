while getopts f: flag
do
	case "${flag}" in
		f) folder_pcds=${OPTARG};;
	esac
done

for outputx in $(ls $folder_pcds)
do
	for categories in $(ls $folder_pcds/input)
	do
		echo $folder_pcds/$outputx/$categories
		ls $folder_pcds/$outputx/$categories/* > results.list
		python3 render_mitsuba2_pc.py results.list 
		rm -rf results.list
		rm -rf $folder_pcds/$outputx/$categories/*.exr
		rm -rf $folder_pcds/$outputx/$categories/*.xml
		rm -rf $folder_pcds/$outputx/$categories/*.pcd
		rm -rf $folder_pcds/$outputx/$categories/*.obj
	done
done

: '
root="../pytorch/benchmark/pcds"
'
