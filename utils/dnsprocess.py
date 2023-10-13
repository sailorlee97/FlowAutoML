# 读取dns.txt
import socket
import dpkt

# from create import walkfile

IP_DNS_DIRECT = {}


def readdnslist(textfile):
    public_dns = set()
    f = open(textfile, mode='r')
    for line in f:
        public_dns.add(line.strip('\n'))
    # print(public_dns)
    f.close()
    return public_dns


# 提取公共服务DNS对应的IP
def extractdnsip(pcapfile, dnset):
    ip_dns_direct = {}
    f = open(pcapfile, mode='rb')
    print('提取DNS对应IP：', pcapfile)

    pkts = dpkt.pcap.Reader(f)
    for ts, buf in pkts:

        eth = dpkt.ethernet.Ethernet(buf)
        # IPv4
        if isinstance(eth.data, dpkt.ip.IP):
            ip = eth.data
            if isinstance(ip.data, dpkt.udp.UDP):
                srcport = ip.data.sport
                dstport = ip.data.dport
                # 判断是否是DNS数据报
                if srcport == 53 or dstport == 53:  # DNS
                    try:
                        dns = dpkt.dns.DNS(ip.data.data)
                    except dpkt.dpkt.UnpackError:
                        continue

                    if dns.qr != 0:
                        for rr in dns.an:
                            if rr.type == dpkt.dns.DNS_A:
                                name = rr.name

                                # 创建字典表示DNS与IP的对应关系
                                for dns in dnset:
                                    if name in dns:
                                        ip = socket.inet_ntop(socket.AF_INET, rr.rdata)
                                        if ip in ip_dns_direct.keys():
                                            dnslist = ip_dns_direct[ip]
                                            dnslist.add(name)
                                            ip_dns_direct[ip] = dnslist
                                        else:
                                            dnslist = set()
                                            dnslist.add(name)
                                            ip_dns_direct[ip] = dnslist

        # IPv6
        elif isinstance(eth.data, dpkt.ip6.IP6):
            ipv6 = eth.data
            if isinstance(ipv6.data, dpkt.udp.UDP):
                srcport = ipv6.data.sport
                dstport = ipv6.data.dport

                # 判断是否是DNS数据报
                if srcport == 53 or dstport == 53:  # DNS
                    try:
                        dns = dpkt.dns.DNS(ipv6.data.data)
                    except dpkt.dpkt.UnpackError:
                        continue

                    if dns.qr != 0:
                        for rr in dns.an:
                            if rr.type == dpkt.dns.DNS_A:
                                name = rr.name

                                # 创建字典表示DNS与IP的对应关系
                                for dns in dnset:
                                    if name in dns:
                                        ip = socket.inet_ntop(socket.AF_INET, rr.rdata)
                                        if ip in ip_dns_direct.keys():
                                            dnslist = ip_dns_direct[ip]
                                            dnslist.add(name)
                                            ip_dns_direct[ip] = dnslist
                                        else:
                                            dnslist = set()
                                            dnslist.add(name)
                                            ip_dns_direct[ip] = dnslist

    f.close()
    return ip_dns_direct


def ipdnsdirect(pcapfile):
    # # 遍历pcap文件
    # pcappath = './pcapfile/'
    # pcappath_list = walkfile(pcappath)

    # 读取公共服务类dns
    global IP_DNS_DIRECT

    # 读取公共服务类dns
    public_dns = readdnslist('./publicdns.txt')

    # # 遍历所有pcap提取dns对应ip
    # for pcapfile in pcappath_list:
    #     # 提取所有dns对应的ip
    #     IP_DNS_DIRECT.update(extractdnsip(pcapfile, public_dns))

    # 提取所有dns对应的ip
    IP_DNS_DIRECT.update(extractdnsip(pcapfile, public_dns))
