
public class FamMember {
    protected String member;
    protected int time;
    FamMember(String member,int time){
        this.member=member;
        this.time=time;
    }
    FamMember(){
        
    }
    public String getMember(){
        return member;
    }
    public int getTime(){
        return time;
    }
    public void setMember(String member){
        this.member=member;
    }
    public void setTime(int time){
        this.time=time;
    }
    
}
