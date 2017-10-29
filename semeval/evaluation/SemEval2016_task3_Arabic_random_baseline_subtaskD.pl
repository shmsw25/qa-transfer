#####################################################
## Generates random baselines from the XML file.   ##
## Threshold value for the percentage of expected  ##
## relevant QA could be specified (default = 20%)  ##
####################################################################################
#Usage: cat collection.xml | perl SemEval2016_task3_Arabic_random_baseline_subtaskD.pl THRESHOLD > baseline.txt #
####################################################################################


$th = $ARGV[0] or $th = 0.2;
$cutoff = $th*30;

while(<STDIN>){
	if(/<Question QID \= "(\d+)"/){
		$qid = $1;
		%score = ();
		%gold = ();
	}
	elsif(/<\/Question>/){
		$r = 1;
		for $qaid (sort {$score{$b} <=> $score{$a}} keys %score){
			$value = "false";
			if($r<=$cutoff){$value = "true";}
			$gold{$qaid} = "$qid\t$qaid\t$r\t$score{$qaid}\t$value";
			$r++;
		}
		for $qaid (sort {$a <=> $b} keys %gold){
			print "$gold{$qaid}\n";
		}
	}
	elsif(/<QApair QAID\="(\d+)" QArel\="[IRD]"/){
		$qaid = $1;
		$score{$qaid} = rand(1);
	}
}
