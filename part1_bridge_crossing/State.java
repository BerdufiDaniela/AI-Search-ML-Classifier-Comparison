import java.util.*; 
public class State implements Comparable<State>
{
	private int f, h, g;
	private State father;
	private int totalTime;
	private ArrayList <String> listmem = new ArrayList<>() ;//lista me me meloi
	
	//constructor - fill with arguments if necessary
	public State(ArrayList<String> listmem) 
	{
		this.listmem=listmem;
		this.f = 0;
		this.h = 0;
		this.g = 0;
		this.father = null;
		this.totalTime = 0;
	}
	//constructor
	public State(){

	}
	public String getelement(int k){
		return getListmem().get(k);
	}

	public void addelement(String s){
		getListmem().add(s);

	}
	public void removel(int d){
		getListmem().remove(d);
	}

	public ArrayList<String> getListmem(){
		return listmem;
	}

	public void setListmem(ArrayList<String> listmem){
		this.listmem=listmem;
	}
	
	// copy constructor
	public State(State s)
	{
		// create a state similar with s...
	}
	
	public int getF() 
	{
		return this.f;
	}
	
	public int getG() 
	{
		return this.g;
	}
	
	public int getH() 
	{
		return this.h;
	}
	
	public State getFather()
	{
		return this.father;
	}
	
	public void setF(int f)
	{
		this.f = f;
	}
	
	public void setG(int g)
	{
		this.g = g;
	}
	
	public void setH(int h)
	{
		this.h = h;
	}
	
	public void setFather(State f)
	{
		this.father = f;
	}
	
	public int getTotalTime() 
	{
		return this.totalTime;
	}
	
	public void setTotalTime(int time)
	{
		this.totalTime = time;
	}
	public void evaluate(FamMember[] fm,int n,int g2) 
	{	
		//dhmiourgw hashmap gia na exw antisoixisw melos me xrono pou kanei gia na perasei thn gefura
		HashMap<String,Integer> listinfo=new HashMap <String,Integer>();
		for(int i =0;i<n;i++){
			listinfo.put(fm[i].getMember(),fm[i].getTime());
		}
		//arxikopoiw xrono
		int time=0;
		//arxikopoiw counter
		int counter=0;
		//o pinakas pou exoume san orisma einai ta3inomimenos se au3ousa seira
		for(int i=n-1;i>=0;i--){
			//arxikopoiw flagsame 
			boolean flagsame=false;
			for(int k=0;k<getListmem().size();k++){
				//sugrinw to stoixeio apo ton pinaka ton arxiko me ola ta meloi me ta meloi pou exoun perasei
				if(fm[i].getMember()==getListmem().get(k)){
					flagsame=true;
				}
			}
			//an to stoixeio den einai kapoio melos apo auta pou exoun perasei
			if(flagsame==false){
				//epeidh 8elw na perasoun 2 meloi ta metraw me to counter
				counter++;
				if(counter==1){
					//brika to 1o melos pou einai to megalhtero kai to pros8etw sto time 
					//epeidh o xronos tou megalhterou metraei
					time=time+listinfo.get(fm[i].getMember());
				}
				//brika kai to 2o megalutero xrono to counter na ginei 0 gia na petrhsv ton megalutero xrono epomenou zeugariou
				if(counter==2){
					counter=0;
				}
			}
		}
		//8etw thn euretikh me ton xrono
		setG(g2);
		setH(time);
		setF(getG()+getH());
		
	}

	public void print(State s) {
		if (s.getListmem().size()==3){
			System.out.printf("The members that should cross the bridge are %s and %s%n",s.getListmem().get(0),s.getListmem().get(1));
			System.out.printf("The member that should return is %s%n",s.getListmem().get(2));
		}
		if(s.getListmem().size()==2){
			System.out.printf("The members that should cross the bridge are %s and %s%n",s.getListmem().get(0),s.getListmem().get(1));
		}
	}

	//sunarthsh pou epistrefei lista antikeimenwn state analoga me thn katastash brisketai twra
	public ArrayList<State> getChildren(FamMember[] fm,int n) {
	
		//dhmiourgw lista me tis katastaseis paidia pou 8a epistrepsw analoga me thn katastash pou briskomai twra
		ArrayList<State> childs=new ArrayList<>();
		//gia ka8e stoixeio tou pinaka me meloi kai xronous
		for (int i=0;i<n;i++){
			//elegxw an to stoixeio pou briskomai twra einai sthn lista ths katastashs mou me ta meloi pou exoun perasei hdh
			boolean flag=false;
			for(int l=0;l<getListmem().size();l++){
				//an to stoixeio sto opoio briskomai twra isoutai me kapoio apo ayta pou exoun perasei
				if(fm[i].getMember()==getListmem().get(l)){
					flag=true;
				}
			}
			//ean den einai ena apo ta meloi pou exoun perasei hdh mporw na dw pi8anous sundoiasmous
			if(flag==false ){
				//gia ka8e epomeno stoixeio
				int j=i+1;
				while(j<n){
					//elegxw an einai idio me kanena melos apo thn katastash me ta meloi pou exoun perasei thn gefura
					boolean flag2=false;
					for(int l=0;l<getListmem().size();l++){
						if(fm[j].getMember()==getListmem().get(l)){
							flag2=true;
						}
					}
					//an den einai 
					if(flag2==false){
						//ftiaxnv mia lista
						State st=new State();
						st.addelement(fm[i].getMember());
						st.addelement(fm[j].getMember());
						//to pros8etw sthn lista childs gia na to epistrepsw
						childs.add(st);
						j++;
					//alliws den kanv tipota proxvrv sto epomeno stoixeio
					}else{
						j++;
					}
				}
			}
	
		}
		//epistefw thn lista me tis pi8anes katastaseis pou mporoun na perasoun apenanti analoga me opoio melos brisketai hdh apenanti
		return childs;
	}
	//elegxei ama einai telikh katastash
	public boolean isFinal(FamMember [] f,int N) {
		//elenxw an h katastash sthn opoia briskomai twra exei ola ta meloi ths oikogeneias
		//an dhladh exoun perasei oloi apenanti
		int counter=0;
		for (int i=0;i<N;i++){
			if(getListmem().contains(f[i].getMember())){
				counter++;
			}
		}
		if (counter==N){
			return true;
		}else{
			return false;
		}
	}
	
	@Override
	public boolean equals(Object obj) {return true;}
	
	@Override
    public int hashCode() {return 0;}
	
	@Override
    public int compareTo(State s)
    {
        return Double.compare(this.f, s.getF()); // compare based on the heuristic score.
    }
}