## Usage: gunzip -c SemEval2016-Task3-CQA-MD-dev-subtaskD.xml.gz | perl SemEval2016_task3_Arabic_generate_relevancy_file_subtaskD.pl > SemEval2016-Task3-CQA-MD-dev-subtaskD.xml.relevancy

while(<>){
	if(/<Question QID \= "(\d+)"/){
		$qid = $1;
		$r = 1;
		%gold = ();
	}
	elsif(/<\/Question>/){
		for $qid (sort {$a <=> $b} keys %gold){
			print "$gold{$qid}";
		}
	}
	elsif(/<QApair QAID\="(\d+)" QArel\="([IRD])"/){
		$qaid = $1;
		$rel = $2;
		$score = 1/$r;
		if($rel eq "I"){
			$gold{$qaid} = "$qid\t$qaid\t$r\t$score\tfalse\n";
		}
		elsif($rel eq "R"){
			$gold{$qaid} = "$qid\t$qaid\t$r\t$score\ttrue\n";
		}
		elsif($rel eq "D"){
			$gold{$qaid} = "$qid\t$qaid\t$r\t$score\ttrue\n";
		}
		$r++;
	}
}
